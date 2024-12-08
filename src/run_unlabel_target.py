# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.append('../')
sys.path.append('../../')
import logging
import argparse
import random
from collections import namedtuple

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, ACT2FN, gelu
from transformers.models.bert.configuration_bert import BertConfig
from optimization import AdamW, Warmup
from data_utils import SATokenizer
import data_utils
import modelconfig


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)



class Intermediate(torch.nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertForClassificationModel(BertPreTrainedModel):
    def __init__(self, config, num_labels=0):
        super(BertForClassificationModel, self).__init__(config)
        self.num_labels = num_labels
        self.config = config
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_feature = Intermediate(config)
        self.sent_classifier = torch.nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, src_input_ids=None, src_attention_mask=None, src_labels=None, tgt_input_ids=None, tgt_attention_mask=None, mode=None):
        if mode == "src_sent":
            _, src_pooled_output = self.bert(src_input_ids, attention_mask=src_attention_mask, output_all_encoded_layers=False,)
            src_pooled_output = self.dropout(src_pooled_output)
            src_sent_feature = self.sentiment_feature(src_pooled_output)
            src_logits = self.sent_classifier(src_sent_feature)
            src_sent_loss = self.CE_loss(src_logits, src_labels)
            return src_sent_loss

        if mode == "tgt_pseudo":
            _, tgt_pooled_output = self.bert(tgt_input_ids, attention_mask=tgt_attention_mask, output_all_encoded_layers=False,)
            tgt_pooled_output = self.dropout(tgt_pooled_output)
            tgt_sent_feature = self.sentiment_feature(tgt_pooled_output)

            cluster_centers = self.sent_classifier.weight
            feat_norm = tgt_sent_feature / tgt_sent_feature.norm(dim=1, keepdim=True)
            center_norm = cluster_centers / cluster_centers.norm(dim=1, keepdim=True)
            cos_sim = torch.mm(feat_norm, center_norm.transpose(0, 1))
            exp_sim = torch.exp(cos_sim / 0.01)
            p = exp_sim / exp_sim.sum(dim=1, keepdim=True)
            loss = -(p * torch.log(p)).sum(dim=1).mean() * 2  # 对每个样本计算p * log(p)，然后对整个批量取平均
            return loss

        if mode == "test":
            _, tgt_pooled_output = self.bert(tgt_input_ids, attention_mask=tgt_attention_mask, output_all_encoded_layers=False,)
            tgt_pooled_output = self.dropout(tgt_pooled_output)
            tgt_sent_feature = self.sentiment_feature(tgt_pooled_output)
            logits = self.sent_classifier(tgt_sent_feature)
            return tgt_sent_feature, logits


def train(args=None, model=None, train_dataset=None, pseudo_labeled_dataset=None, eval_dataset=None):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    pseudo_labeled_dataloader = DataLoader(pseudo_labeled_dataset, sampler=RandomSampler(pseudo_labeled_dataset), batch_size=args.train_batch_size)
    num_train_steps = int(len(train_dataloader)) * args.num_train_epochs

    def creatOptimizer(named_parameters, num_train_steps=num_train_steps):
        param_optimizer = [(k, v) for k, v in named_parameters if v.requires_grad == True]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.warmup_proportion)
        scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)
        return optimizer, scheduler

    def steps(loss, optimizer, scheduler, model):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    model.cuda()
    optimizer, scheduler = creatOptimizer(list(model.bert.named_parameters()) + list(model.sentiment_feature.named_parameters()) + list(model.sent_classifier.named_parameters()))

    logger.info(" Batch size = %d", args.train_batch_size)
    logger.info("Total optimization steps = %d", num_train_steps)
    model.zero_grad()
    model.train()
    global_step = 0
    loss1, loss2 = torch.tensor([0]), torch.tensor([0])
    for epoch in range(int(args.num_train_epochs)):
        target_dataloader = iter(pseudo_labeled_dataloader)
        for step, batch in enumerate(train_dataloader):
            src_batch = tuple(t.to(args.device) for t in batch)
            src_input_ids, src_input_mask, src_label_ids = src_batch
            loss1 = model(src_input_ids=src_input_ids, src_attention_mask=src_input_mask, src_labels=src_label_ids, mode="src_sent")
            steps(loss1, optimizer, scheduler, model)
            if (global_step + 1 ) % 30 == 0:
                model.sent_classifier.weight.requires_grad = False
                tgt_batch = tuple(t.to(args.device) for t in target_dataloader.next())
                tgt_input_ids, tgt_input_mask, _ = tgt_batch
                loss2 = model(tgt_input_ids=tgt_input_ids, tgt_attention_mask=tgt_input_mask, mode="tgt_pseudo")
                steps(loss2, optimizer, scheduler, model)
                model.sent_classifier.weight.requires_grad = True

            global_step += 1
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, {:.5f}".format(epoch, global_step, num_train_steps, loss1.item(), loss2.item()))
        

def pseudo_label_of_unlabeled_data(args, model=None, dataset=None):
    model.eval()
    target_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    pooled_outputs = []
    for _, batch in enumerate(target_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, _ = batch
        with torch.no_grad():
            pooled_output, _ = model(tgt_input_ids=input_ids, tgt_attention_mask=input_mask, mode="test")
            pooled_outputs.append(pooled_output)
    pooled_outputs_pt = torch.cat(pooled_outputs, dim=0)
    logger.info("Computing pseudo label from dataset file {}.".format(pooled_outputs_pt.shape))
    model.train()
    return pooled_outputs_pt


def evaluate(args=None, model=None, eval_dataset=None):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.eval_batch_size)
    model.eval()

    out_preds, out_labes, out_logts = [], [], []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, label_ids = batch
            _, logits = model(tgt_input_ids=input_ids, tgt_attention_mask=input_mask, mode="test")
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            out_preds.append(predicts.detach().cpu().numpy())
            out_labes.append(label_ids.detach().cpu().numpy())
            out_logts.append(logits.detach().cpu().numpy())
    y_true = np.concatenate(tuple(out_labes), axis=0)
    y_pred = np.concatenate(tuple(out_preds), axis=0)
    y_logt = np.concatenate(tuple(out_logts), axis=0)
    
    acc = accuracy_score(y_true, y_pred)
    logger.info("accuracy: {:.4}".format(acc))
    np.savez(os.path.join(args.output_dir, "predicts_labels.npz"), y_true=y_true, y_pred=y_pred, y_logt=y_logt)


def load_and_cache_examples(args=None, dataname="train"):
    processor = data_utils.SAProcessor()
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)
        logger.info("********** Running training **********")
        logger.info(" Num examples = %d", len(examples))
    elif dataname == "test":
        examples = processor.get_test_examples(args.data_dir)
        logger.info("********** Running evaluation **********")
        logger.info(" Num examples = %d", len(examples))
        logger.info(" Batch size = %d", args.eval_batch_size)
    elif dataname == "unlabeled":
        examples = processor.get_target_unexamples(args.data_dir)
        examples = examples[:4000]
        logger.info("********** Running unlabeled **********")
        logger.info(" Num examples = %d", len(examples))
    else:
        raise ValueError("(evaluate and dataname) parameters error !")

    features = data_utils.convert_examples_to_features(examples, args.max_seq_length, args.tokenizer, printN=0,)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert_base', type=str)
    parser.add_argument("--data_dir", default='../data/amazon/books-dvs', type=str, required=False, help="The input data dir containing json files.")
    parser.add_argument('--task_type', default='sa', type=str, help="random seed for initialization")
    parser.add_argument("--output_dir", default='../response/amazon/books-dvs', type=str, required=False,
        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument('--eval_logging_steps', type=int, default=100, help="Log every X evalution steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")

    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, 
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    Instance = SATokenizer(bert_name=args.bert_model, archive=modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    args.tokenizer = Instance.tokenizer

    processor = data_utils.SAProcessor()
    label_list = processor.get_labels()

    train_dataset = load_and_cache_examples(args=args, dataname="train")
    unlabeled_dataset = load_and_cache_examples(args=args, dataname="unlabeled")
    test_dataset = load_and_cache_examples(args=args, dataname="test")

    config = BertConfig.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    config.update({
        'output_dir': args.output_dir,
        'max_seq_length': args.max_seq_length,
        })
    model = BertForClassificationModel.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], config=config, num_labels=len(label_list))
    model.cuda()

    if args.do_train:
        train(args=args, model=model, train_dataset=train_dataset, pseudo_labeled_dataset=unlabeled_dataset, eval_dataset=test_dataset)
    if args.do_eval:
        evaluate(args=args, model=model, eval_dataset=test_dataset)



if __name__ == "__main__":
    main()
