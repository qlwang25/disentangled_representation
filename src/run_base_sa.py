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
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from optimization import AdamW, Warmup
from data_utils import SATokenizer
import data_utils
import modelconfig

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()
        return output


class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)
        self.domain_classifier = torch.nn.Linear(config.hidden_size, 2)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        sequence_output, pooled_output = outputs 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss
        else:
            return pooled_output, logits

    def domain(self, src_inputs, src_masks, tgt_inputs, tgt_masks):
        inputs = torch.cat([src_inputs[:8, :], tgt_inputs[:8, :]], dim=0)
        masks = torch.cat([src_masks[:8, :], tgt_masks[:8, :]], dim=0)
        _, outputs = self.bert(inputs, attention_mask=masks, output_all_encoded_layers=False)
        outputs = self.dropout(outputs)
        domain_output = ReverseLayerF.apply(outputs)
        logit = self.domain_classifier(domain_output)
        loss = self.loss_fct(logit, torch.cat([torch.ones(src_inputs[:8, :].size(0)).long(), torch.ones(tgt_inputs[:8, :].size(0)).long()], dim=0).cuda())
        return loss


def train(args, trainset, testset, unlabelset, model):
    train_dataloader = DataLoader(trainset, sampler=RandomSampler(trainset), batch_size=args.train_batch_size)
    unlabel_dataloader = DataLoader(unlabelset, sampler=RandomSampler(unlabelset), batch_size=args.train_batch_size)
    
    num_train_steps = int(len(train_dataloader)) * args.num_train_epochs
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.warmup_proportion)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)

    logger.info("Total optimization steps = %d", num_train_steps)
    model.zero_grad()
    model.train()
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        target_dataloader = iter(unlabel_dataloader)
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, labels = batch
            loss = model(inputs, attention_mask=masks, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if global_step % 2 == 0:
                try:
                    tgt_batch = target_dataloader.next()
                except Exception as e:
                    target_dataloader = iter(unlabel_dataloader)
                    tgt_batch = target_dataloader.next()
                tgt_batch = tuple(t.to(args.device) for t in tgt_batch)
                tgt_inputs, tgt_masks, _ = tgt_batch
                domain_loss = model.domain(src_inputs=inputs, src_masks=masks, tgt_inputs=tgt_inputs, tgt_masks=tgt_masks)
                domain_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            global_step += 1
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, Domain Loss:{:.5f}".format(epoch, global_step, num_train_steps, loss.item(), domain_loss.item()),)           
                if args.evaluate_during_training and global_step % args.eval_logging_steps == 0: 
                    model.eval()
                    evaluate(args, testset=testset, model=model)
                    model.train()
        torch.cuda.empty_cache()


def evaluate(args, testset, model, domain='target'):
    eval_dataloader = DataLoader(testset, sampler=SequentialSampler(testset), batch_size=args.eval_batch_size)
    out_preds, out_labes = [], []
    out_hiddens, out_sents = [], []
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
            pooled_output, logits = model(input_ids, attention_mask=input_mask)
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            out_preds.append(predicts.detach().cpu().numpy())
            out_labes.append(label_ids.detach().cpu().numpy())
            out_hiddens.append(pooled_output)
            out_sents.append(label_ids)
        torch.cuda.empty_cache()
    
    y_true = np.concatenate(tuple(out_labes), axis=0)
    y_pred = np.concatenate(tuple(out_preds), axis=0)
    logger.info("accuracy: {:.4}; precision:{:.4}; recall:{:.4}; f1:{:.4}".format(accuracy_score(y_true, y_pred), 
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro'),
        ),
    )


def load_and_cache_examples(args, dataname="train"):
    processor = data_utils.SAProcessor()
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)
        examples = examples[:args.labeled_num]
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", args.train_batch_size)

    elif dataname == "dev":
        examples = processor.get_dev_examples(args.data_dir)
        logger.info("***** Running validations *****")
        logger.info(" Num orig examples = %d", len(examples))

    elif dataname == "test":
        examples = processor.get_test_examples(args.data_dir)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

    elif dataname == "retrieval":
        examples = processor.get_target_unexamples(args.data_dir)
        examples = examples[:args.unlabeled_num]
        logger.info("***** Running retrieval *****")
        logger.info("  Num examples = %d", len(examples))

    else:
        raise ValueError("(evaluate and dataname) parameters error !")
    
    features = data_utils.convert_examples_to_features(examples, args.max_seq_length, args.tokenizer, printN=1,)
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
    parser.add_argument("--output_dir", default='../run_out/amazon/books-dvs', type=str, required=False,
        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,
        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'` , `'none'`,"
                         " `None`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
    parser.add_argument('--eval_logging_steps', type=int, default=200, help="Log every X evalution steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")

    parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_valid", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=4e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, 
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--labeled_num", default=1600, type=int, help="how much labeled sample")
    parser.add_argument("--unlabeled_num", default=4000, type=int, help="how much unlabeled sample")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    Instance = SATokenizer(bert_name='bert_base', archive=modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    args.tokenizer = Instance.tokenizer
    args.bert_name = Instance.name

    processor = data_utils.SAProcessor()
    label_list = processor.get_labels()
    model = BertForClassification.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], num_labels=len(label_list))
    model.cuda()

    dataset = load_and_cache_examples(args, dataname="train")
    undataset = load_and_cache_examples(args, dataname="retrieval")
    dev_data = load_and_cache_examples(args, dataname="dev")
    eval_data = load_and_cache_examples(args, dataname="test")

    if args.do_train:
        train(args, trainset=dataset, testset=dev_data, unlabelset=undataset, model=model)
    if args.do_eval:
        evaluate(args, testset=dataset, model=model, domain='source')
        evaluate(args, testset=eval_data, model=model, domain='target')


if __name__ == "__main__":
    main()