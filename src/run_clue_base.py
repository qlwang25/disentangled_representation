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

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertSelfAttention
from transformers.models.bert.configuration_bert import BertConfig
from optimization import AdamW, Warmup
from data_utils import SATokenizer
import data_utils
import modelconfig


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class MuIn_loss(torch.nn.Module):
    def __init__(self, MI_threshold):
        super(MuIn_loss, self).__init__()
        self.MI_threshold = MI_threshold

    def forward(self, logits):
        p = torch.nn.functional.softmax(logits, dim=-1)
        condi_entropy = -torch.sum(p * torch.log(p), dim=-1).mean()
        y_dis = torch.mean(p, dim=0)
        y_entropy = (-y_dis * torch.log(y_dis)).sum()
        if y_entropy.item() < self.MI_threshold:
            return -y_entropy + condi_entropy, y_entropy
        else:
            return condi_entropy, y_entropy        


class BertForClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=0):
        super(BertForClassification, self).__init__(config)
        self.num_labels = num_labels
        self.config = config
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        if hasattr(config, 'retrieval_num'):
            self.clue_embeddings = torch.nn.Embedding(2000, config.hidden_size).requires_grad_(False)
            self.clue_embeddings.weight.data.copy_(torch.load(config.output_dir + '/' + 'source_data_repres.pt').cuda())
            self.self = BertSelfAttention(config, output_attentions=False, keep_multihead_output=False)
    
        self.KL_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.MI_loss = MuIn_loss(MI_threshold=0.5)
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, clue_ids=None, senti_type_ids=None, pseudo_labels=None, labels=None,):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False,)
        sequence_output, pooled_output = outputs 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        batch_size, dim = pooled_output.size()
        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss

        elif pseudo_labels is not None:
            ipt = torch.nn.functional.log_softmax(logits, dim=1)
            kl_loss = self.KL_loss(ipt, pseudo_labels)
            # mi_loss, _ = self.MI_loss(logits)

            if hasattr(self.config, 'retrieval_num'):
                context_ipt = torch.cat([pooled_output.unsqueeze(1), self.clue_embeddings(clue_ids)], dim=1)
                context_mask = torch.ones((batch_size, self.config.retrieval_num + 1), dtype=torch.long, device=pooled_output.device).unsqueeze(1).unsqueeze(2)
                self_output = self.self(hidden_states=context_ipt, attention_mask=context_mask, head_mask=None)
                self_output = self.dropout(self_output)
                target_output = self_output[:, 0, :]
                context_probs = torch.nn.functional.log_softmax(self.classifier(target_output), dim=1)
                context_loss = self.KL_loss(context_probs, pseudo_labels)
                kl_loss += 0.1 * context_loss                

                source_output = self_output[:, 1:, :]
                source_logits = self.classifier(source_output)
                source_loss = self.CE_loss(source_logits.view(-1, 2), senti_type_ids.view(-1))
                kl_loss += 0.1 * source_loss
            return kl_loss

        else:
            return pooled_output, logits


def train(args=None, model=None, train_dataset=None, eval_dataset=None):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    num_train_steps = int(len(train_dataloader)) * args.num_train_epochs

    model.cuda()
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
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, _, clue_ids, senti_type, pseudo_labels = batch
            loss = model(input_ids=input_ids, attention_mask=input_mask, clue_ids=clue_ids, senti_type_ids=senti_type, pseudo_labels=pseudo_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}".format(epoch, 
                    global_step, 
                    num_train_steps,
                    loss.item()),
                )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                global_step += 1
                if args.evaluate_during_training and global_step % args.eval_logging_steps == 0: 
                    model.eval()
                    evaluate(args=args, model=model, eval_dataset=eval_dataset, dataname="test")
                    model.train()
        torch.cuda.empty_cache()


def retrieval(args, source_dataname=None, target_dataname=None, model=None):
    model.eval()

    source_data = load_and_cache_examples(args=args, base_feature=True, dataname=source_dataname)
    source_dataloader = DataLoader(source_data, sampler=SequentialSampler(source_data), batch_size=args.eval_batch_size)
    out_repres, out_labels = [], []
    for _, batch in enumerate(source_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
            pooled_output, _ = model(input_ids, attention_mask=input_mask)
            out_repres.append(pooled_output)
            out_labels.append(label_ids)
        torch.cuda.empty_cache()
    out_repres_spt = torch.cat(out_repres, dim=0)
    out_labels_spt = torch.cat(out_labels, dim=0)
    torch.save(out_repres_spt, args.output_dir + '/' + 'source_data_repres.pt')
    torch.save(out_labels_spt, args.output_dir + '/' + 'source_data_labels.pt')
    logger.info("Saving features {} and labels {} from source dataset file.".format(out_repres_spt.shape, out_labels_spt.shape))

    target_data = load_and_cache_examples(args=args, base_feature=True, dataname=target_dataname)
    target_dataloader = DataLoader(target_data, sampler=SequentialSampler(target_data), batch_size=args.eval_batch_size)
    out_repres, out_preds = [], []
    for _, batch in enumerate(target_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, _ = batch
        with torch.no_grad():
            pooled_output, logits = model(input_ids, attention_mask=input_mask)
            out_repres.append(pooled_output)
            out_preds.append(torch.nn.Softmax(dim=1)(logits))
        torch.cuda.empty_cache()
    out_repres_tpt = torch.cat(out_repres, dim=0)
    out_preds_tpt = torch.cat(out_preds, dim=0)
    logger.info("Computing features {} from target dataset file.".format(out_repres_tpt.shape))

    indices, clue_labels = [], []
    for out_repre_tpt in torch.split(out_repres_tpt, 1000):
        indices_tmp, clue_labels_tmp = compute_clues(args=args, query_tensor=out_repre_tpt, key_tensor=out_repres_spt, value_tensor=out_labels_spt)
        indices.append(indices_tmp)
        clue_labels.append(clue_labels_tmp)
    indices =  torch.cat(indices, dim=0)
    clue_labels =  torch.cat(clue_labels, dim=0)
    return indices, clue_labels, out_preds_tpt


def evaluate(args=None, model=None, eval_dataset=None, dataname="test"):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.eval_batch_size)

    model.eval()
    out_preds, out_labes = [], []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, label_ids, _, _, _ = batch
            _, logits = model(input_ids=input_ids, attention_mask=input_mask)
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            out_preds.append(predicts.detach().cpu().numpy())
            out_labes.append(label_ids.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    y_true = np.concatenate(tuple(out_labes), axis=0)
    y_pred = np.concatenate(tuple(out_preds), axis=0)
    logger.info("accuracy: {:.4}; precision:{:.4}; recall:{:.4}; f1:{:.4}".format(accuracy_score(y_true, y_pred), 
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro'),
        ),
    )


def compute_clues(args, query_tensor=None, key_tensor=None, value_tensor=None):
    batch_size, _ = query_tensor.size()
    retrieval_N, _ = key_tensor.size()
    # dot = torch.matmul(query_tensor, torch.transpose(key_tensor, dim0=0, dim1=1))

    ipt1 = query_tensor.unsqueeze(1).repeat(1, retrieval_N, 1)
    ipt2 = key_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
    dot = torch.nn.CosineSimilarity(dim=2, eps=1e-6)(ipt1, ipt2)
    _, indices = torch.topk(dot, args.retrieval_num, dim=1, largest=True, sorted=True)
    clue_labels = torch.index_select(value_tensor, dim=0, index=indices.view(-1))
    clue_labels = clue_labels.view(batch_size, args.retrieval_num)
    return indices, clue_labels


def load_and_cache_examples(args=None, trained_model=None, base_feature=False, dataname="train"):
    processor = data_utils.SAProcessor()
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)
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

    elif dataname == "unlabeled":
        examples = processor.get_target_unexamples(args.data_dir)
        examples = examples[:6000]
        logger.info("***** Running unlabeled *****")
        logger.info("  Num examples = %d", len(examples))

    else:
        raise ValueError("(evaluate and dataname) parameters error !")

    if base_feature:
        features = data_utils.convert_examples_to_features(examples, args.max_seq_length, args.tokenizer, printN=0,)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        return dataset

    if not base_feature:
        logger.info("............. Retrieval ing ...............")
        if dataname == "unlabeled":
            indices, clue_labels, pseudo_labels = retrieval(args=args, source_dataname="train", target_dataname="unlabeled", model=trained_model)
        if dataname == "test":
            indices, clue_labels, pseudo_labels = retrieval(args=args, source_dataname="train", target_dataname="test", model=trained_model)
        logger.info("............. Retrieval Done ...............")

        features = data_utils.clue_based_convert_examples_to_features(
            examples=examples, 
            labeled_examples=None, ## rouge & bleu
            max_seq_length=args.max_seq_length,
            retrieval_num=args.retrieval_num,
            indices=indices,
            clue_labels=clue_labels,
            pseudo_labels=pseudo_labels,
            tokenizer=args.tokenizer,
            )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_clue_ids = torch.tensor([f.clue_ids for f in features], dtype=torch.long)
        all_senti_type = torch.tensor([f.senti_type for f in features], dtype=torch.long)
        all_pseudo_labels = torch.tensor([f.pseudo_label for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_clue_ids, all_senti_type, all_pseudo_labels)
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
    parser.add_argument('--eval_logging_steps', type=int, default=100, help="Log every X evalution steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")

    parser.add_argument("--max_seq_length", default=512, type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
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
    parser.add_argument("--do_retrieval", default=False, action='store_true', help="Whether to retrieval other sentence.")
    parser.add_argument("--retrieval_num", default=3, type=int, help="how much retrieval sample")
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
    trained_model = BertForClassification.from_pretrained(args.output_dir, num_labels=len(label_list))
    trained_model.cuda()
    trained_model.eval()

    train_dataset = load_and_cache_examples(args=args, trained_model=trained_model, base_feature=False, dataname="unlabeled")
    dev_dataset = load_and_cache_examples(args=args, trained_model=trained_model, base_feature=False, dataname="dev")
    eval_dataset = load_and_cache_examples(args=args, trained_model=trained_model, base_feature=False, dataname="test")

    config = BertConfig.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    config.update({
        'retrieval_num':args.retrieval_num, 
        'output_dir': args.output_dir,
        'max_seq_length': args.max_seq_length,
        })
    model = BertForClassification.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], config=config, num_labels=len(label_list))
    model.cuda()

    if args.do_train:
        train(args=args, model=model, train_dataset=train_dataset, eval_dataset=dev_dataset)
    if args.do_eval:
        evaluate(args=args, model=model, eval_dataset=eval_dataset, dataname="test")


if __name__ == "__main__":
    main()