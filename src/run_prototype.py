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
# os.environ["CUDA_VISIBLE_DEVICES"] = 0

import sys
sys.path.append('../')
sys.path.append('../../')
import logging
import argparse
import random

import torch
import numpy as np
import torch.nn.functional as FUNS
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

from transformers.models.bert.modeling_bert import ACT2FN, gelu
from optimization import AdamW, Warmup
from data_utils import SATokenizer
import data_utils
import modelconfig
from run_base_sa import BertForClassification

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

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


class BertForDomain(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForDomain, self).__init__(config)
        self.bert = BertModel(config, output_attentions=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.domain_classifier = torch.nn.Linear(config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        return pooled_output


class BertaForClassificationModel(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, *model_args, **model_kwargs):
        super(BertaForClassificationModel, self).__init__(config)
        self.bert = BertModel(config, output_attentions=False)
        self.config = config
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_feature = Intermediate(config)
        self.sent_classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        # one text encoder 
        # self.domain_feature = Intermediate(config)
        # self.domain_classifier = torch.nn.Linear(config.hidden_size, 2)
        # self.sent_embedding = torch.nn.Embedding(2, config.hidden_size) # copy weight from self.sentiment_feature

        # two text encoder 
        self.domain = BertForDomain(config)

        self.KL_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.CE_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.MSE_loss = torch.nn.MSELoss(reduction='mean')
        self.sim = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        self.softmax = torch.nn.Softmax(dim=-1)

    def orthogonal_project(self, feature1, feature2, eps=1e-08):
        out1 = torch.div(feature1 * feature2, FUNS.normalize(feature2, p=2.0, dim=1)) 
        out2 = torch.div(feature2, FUNS.normalize(feature2, p=2.0, dim=1))
        return out1 * out2

    def contrastive_loss(self, instances, proto=None, tao=0.05, proto_labels=None):
        norm_features = FUNS.normalize(instances, p=2.0, dim=1)
        norm_proto = FUNS.normalize(proto, p=2.0, dim=-1)
        cosim = torch.matmul(norm_features, norm_proto.t())
        if proto_labels.dim() == 2:
            p = self.softmax(cosim / tao)
            loss = -torch.mean(p * torch.log(p))
        if proto_labels.dim() == 1:
            loss = self.CE_loss(cosim, proto_labels)
        return loss

    def s2s_contrastive_learning(self, features, labels, tao=0.05):
        batch_size, dim = features.size()
        norm_features = FUNS.normalize(features, p=2.0, dim=1)
        matrix = torch.matmul(norm_features, norm_features.t())
        matrix = matrix / tao
        exp_matrix = torch.exp(matrix)
        mask = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        exp_matrix = exp_matrix * mask.cuda()
        denominator = torch.sum(exp_matrix, dim=1)

        unmask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).long()
        unmask_matrix = unmask * exp_matrix
        molecule = torch.sum(unmask_matrix, dim=1)
        divr = torch.div(molecule, denominator)
        divr = torch.masked_select(divr, torch.ne(divr, 0))
        loss = -torch.log(divr).mean()
        return loss

    def t2s_prototype_learning(self, src_features, src_labels, tgt_features, tao=0.05):
        batch_size, dim = src_features.size()
        def obtain_loss(embeds, features):
            if embeds.nelement() == 0:
                embeds = embeds.view(-1, dim).mean(0)
                norm_features = FUNS.normalize(features, p=2.0, dim=1)
                embeds = FUNS.normalize(embeds, p=2.0, dim=0)
                score = torch.matmul(norm_features, embeds)
                score = score / tao
                p = self.softmax(score)
                loss = -torch.mean(p * torch.log(p))
                return loss
            else:
                return torch.zeros(1).cuda()

        neg_embeds = self.sent_embedding.weight[0]
        pos_embeds = self.sent_embedding.weight[1]
        neg_loss = obtain_loss(embeds=neg_embeds, features=tgt_features)
        pos_loss = obtain_loss(embeds=pos_embeds, features=tgt_features)
        return neg_loss + pos_loss

    def s2e_t2e_e2e_prototype_learning(self, src_features, src_labels, tgt_features, tgt_labels, tao=0.05):
        s2e_loss = self.contrastive_loss(instances=src_features, proto=self.sent_embedding.weight, tao=tao, proto_labels=src_labels)
        t2e_loss = self.contrastive_loss(instances=tgt_features, proto=self.sent_embedding.weight, tao=tao, proto_labels=tgt_labels)
        norm_sent_embed = FUNS.normalize(self.sent_embedding.weight, p=2.0, dim=1)
        dot = torch.matmul(norm_sent_embed, norm_sent_embed.t())
        e2e_loss = self.MSE_loss(input=dot, target=torch.eye(2).cuda())
        return s2e_loss, t2e_loss, e2e_loss

    def forward(self, src_input_ids=None, src_attention_mask=None, src_labels=None, tgt_input_ids=None, tgt_attention_mask=None, pseudo_labels=None, KMeans=None):
        if src_input_ids is not None and tgt_input_ids is not None and src_labels is None:
            # domain loss
            # one text encoder 
            # _, src_pooled_output = self.bert(src_input_ids, attention_mask=src_attention_mask, output_all_encoded_layers=False,)
            # src_pooled_output = self.dropout(src_pooled_output)
            # _, tgt_pooled_output = self.bert(tgt_input_ids, attention_mask=tgt_attention_mask, output_all_encoded_layers=False,)
            # tgt_pooled_output = self.dropout(tgt_pooled_output)
            # src_domain_feature = self.domain_feature(src_pooled_output)
            # tgt_domain_feature = self.domain_feature(tgt_pooled_output)

            # two text encoder
            src_batch_size, _ = src_input_ids.size()
            tgt_batch_size, _ = tgt_input_ids.size()
            src_domain_feature = self.domain(input_ids=src_input_ids, attention_mask=src_attention_mask)
            tgt_domain_feature = self.domain(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask)

            src_domain_loss = self.CE_loss(self.domain.domain_classifier(src_domain_feature), torch.zeros(src_batch_size, dtype=torch.long).cuda())
            tgt_domain_loss = self.CE_loss(self.domain.domain_classifier(tgt_domain_feature), torch.ones(tgt_batch_size, dtype=torch.long).cuda())
            return torch.tensor([0]), src_domain_loss + tgt_domain_loss, torch.tensor([0]), torch.tensor([0])

        if src_labels is not None and pseudo_labels is not None:
            src_batch_size, _ = src_input_ids.size()
            tgt_batch_size, _ = tgt_input_ids.size()

            _, src_pooled_output = self.bert(src_input_ids, attention_mask=src_attention_mask)
            src_pooled_output = self.dropout(src_pooled_output)
            _, tgt_pooled_output = self.bert(tgt_input_ids, attention_mask=tgt_attention_mask)
            tgt_pooled_output = self.dropout(tgt_pooled_output)

            # source sentiment loss    
            src_sent_feature = self.sentiment_feature(src_pooled_output)
            src_sent_loss = self.CE_loss(self.sent_classifier(src_sent_feature), src_labels)

            tgt_sent_feature = self.sentiment_feature(tgt_pooled_output)
            # tgt_sent_loss = self.KL_loss(FUNS.log_softmax(self.sent_classifier(tgt_sent_feature), dim=1), pseudo_labels)


            # one text encoder 
            # src_op_domain_loss = self.KL_loss(FUNS.log_softmax(self.domain_classifier(src_sent_feature), dim=1), torch.full((src_batch_size, 2), fill_value=0.5, dtype=torch.float).cuda())
            # tgt_op_domain_loss = self.KL_loss(FUNS.log_softmax(self.domain_classifier(tgt_sent_feature), dim=1), torch.full((tgt_batch_size, 2), fill_value=0.5, dtype=torch.float).cuda())
            # two text encoder
            src_op_domain_loss = self.KL_loss(FUNS.log_softmax(self.domain.domain_classifier(src_sent_feature), dim=1), torch.full((src_batch_size, 2), fill_value=0.5, dtype=torch.float).cuda())
            tgt_op_domain_loss = self.KL_loss(FUNS.log_softmax(self.domain.domain_classifier(tgt_sent_feature), dim=1), torch.full((tgt_batch_size, 2), fill_value=0.5, dtype=torch.float).cuda())

            tgt_sim_loss = self.contrastive_loss(instances=tgt_sent_feature, proto=self.sent_classifier.weight, tao=0.05, proto_labels=pseudo_labels)
            if KMeans is None:
                return src_sent_loss, torch.tensor([0]), 0.1*(src_op_domain_loss+tgt_op_domain_loss), tgt_sim_loss

        else:
            _, src_pooled_output = self.bert(src_input_ids, attention_mask=src_attention_mask)
            src_pooled_output = self.dropout(src_pooled_output)
            src_sent_feature = self.sentiment_feature(src_pooled_output)
            src_domain_feature = self.domain(input_ids=src_input_ids, attention_mask=src_attention_mask)
            logits = self.sent_classifier(src_sent_feature)
            return src_pooled_output, src_sent_feature, src_domain_feature, logits


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
    sent_optimizer, sent_scheduler = creatOptimizer(list(model.bert.named_parameters()) + list(model.sentiment_feature.named_parameters()) + list(model.sent_classifier.named_parameters()))
    # domain_optimizer, domain_scheduler = creatOptimizer(list(model.bert.named_parameters()) + list(model.domain_feature.named_parameters()) + list(model.domain_classifier.named_parameters()))
    # two text encoder
    domain_optimizer, domain_scheduler = creatOptimizer(model.domain.named_parameters())

    logger.info(" Batch size = %d", args.train_batch_size)
    logger.info("Total optimization steps = %d", num_train_steps)
    model.zero_grad()
    model.train()
    global_step = 0
    max_acc = -1
    kmeans = None
    for epoch in range(int(args.num_train_epochs)):
        target_dataloader = iter(pseudo_labeled_dataloader)
        for step, batch in enumerate(train_dataloader):
            src_batch = tuple(t.to(args.device) for t in batch)
            src_input_ids, src_input_mask, src_label_ids = src_batch
            try:
                tgt_batch = target_dataloader.next()
            except Exception as e:
                target_dataloader = iter(pseudo_labeled_dataloader)
                tgt_batch = target_dataloader.next()
            tgt_batch = tuple(t.to(args.device) for t in tgt_batch)
            tgt_input_ids, tgt_input_mask, tgt_psulabels = tgt_batch

            if epoch < 3:
                loss1, loss2, loss3, loss4 = model(src_input_ids=src_input_ids,
                    src_attention_mask=src_input_mask,
                    tgt_input_ids=tgt_input_ids,
                    tgt_attention_mask=tgt_input_mask,
                    )
                steps(loss2, domain_optimizer, domain_scheduler, model)

            else:
                for param in model.domain.parameters():
                    param.requires_grad = False

                loss1, loss2, loss3, loss4 = model(src_input_ids=src_input_ids, 
                    src_attention_mask=src_input_mask, 
                    src_labels=src_label_ids, 
                    tgt_input_ids=tgt_input_ids, 
                    tgt_attention_mask=tgt_input_mask, 
                    pseudo_labels=tgt_psulabels,
                    KMeans=kmeans,
                    )
                steps(loss1 + loss3 + loss4, sent_optimizer, sent_scheduler, model)
    
            global_step += 1
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}, {:.5f}, {:.5f}, {:.5f}".format(epoch, global_step, num_train_steps, 
                    loss1.item(), loss2.item(), loss3.item(), loss4.item()))

                if args.evaluate_during_training and global_step % args.eval_logging_steps == 0 and epoch >= 3: 
                    model.eval()
                    evaluate(args=args, model=model, eval_dataset=eval_dataset, dataname="test")
                    model.train()
        torch.cuda.empty_cache()
        
        if False:
            model.eval()
            unlabeled_dataset = load_and_cache_examples(args=args, trained_model=model, base_feature=False, bert_name='bert', dataname="unlabeled", printN=0)
            pseudo_labeled_dataloader = DataLoader(unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset), batch_size=args.train_batch_size)
            model.train()

        if False:
            model.eval()
            pooled_outputs_spt, _ = pseudo_label_to_unlabeled_data(args, dataname="train", model=model, dataset=train_dataset)
            kmeans = KMeans(n_clusters=2, max_iter=500, random_state=0).fit(pooled_outputs_spt.detach().cpu().numpy())
            model.train()

    logger.info("data dir : {}".format(args.data_dir))
    logger.info(" ACC = {}".format(max_acc))


def pseudo_label_to_unlabeled_data(args, dataname=None, model=None, dataset=None):
    model.eval()
    target_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    pooled_outputs, out_preds = [], []
    for _, batch in enumerate(target_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, _ = batch
        with torch.no_grad():
            pooled_output, logits = model(input_ids, input_mask)
            pooled_outputs.append(pooled_output)
            out_preds.append(torch.nn.Softmax(dim=1)(logits))
        torch.cuda.empty_cache()
    pooled_outputs_pt = torch.cat(pooled_outputs, dim=0)
    out_preds_tpt = torch.cat(out_preds, dim=0)
    logger.info("Computing pseudo label {} from target dataset file {}.".format(out_preds_tpt.shape, pooled_outputs_pt.shape))
    return pooled_outputs_pt, out_preds_tpt


def evaluate(args=None, model=None, eval_dataset=None, dataname="test", domain='target'):
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.eval_batch_size)
    model.eval()
    out_preds, out_labes = [], []
    out_texts, out_domains, out_hiddens, out_sents = [], [], [], []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, label_ids = batch
            src_pooled_output, src_sent_feature, src_domain_feature, logits = model(src_input_ids=input_ids, src_attention_mask=input_mask)
            probability = torch.nn.Softmax(dim=1)(logits)
            predicts = torch.max(probability, dim=1)[1]
            out_preds.append(predicts.detach().cpu().numpy())
            out_labes.append(label_ids.detach().cpu().numpy())
            out_texts.append(src_pooled_output)
            out_domains.append(src_domain_feature)
            out_hiddens.append(src_sent_feature)
            out_sents.append(label_ids)
        torch.cuda.empty_cache()
    
    y_true = np.concatenate(tuple(out_labes), axis=0)
    y_pred = np.concatenate(tuple(out_preds), axis=0)
    acc = accuracy_score(y_true, y_pred)
    logger.info("accuracy: {:.4}; precision:{:.4}; recall:{:.4}; f1:{:.4}".format(acc, 
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='macro'),
        ),
    )


def load_and_cache_examples(args=None, trained_model=None, base_feature=False, bert_name='bert', dataname="train", printN=1):
    processor = data_utils.SAProcessor()
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if dataname == "train":
        examples = processor.get_train_examples(args.data_dir)
        examples = examples[:args.labeled_num]
        logger.info("********** Running training **********")
        logger.info(" Num examples = %d", len(examples))

    elif dataname == "dev":
        examples = processor.get_dev_examples(args.data_dir)
        logger.info("********** Running validations **********")
        logger.info(" Num orig examples = %d", len(examples))

    elif dataname == "test":
        examples = processor.get_test_examples(args.data_dir)
        logger.info("********** Running evaluation **********")
        logger.info(" Num examples = %d", len(examples))
        logger.info(" Batch size = %d", args.eval_batch_size)

    elif dataname == "unlabeled":
        examples = processor.get_target_unexamples(args.data_dir)
        examples = examples[:args.unlabeled_num]
        logger.info("********** Running unlabeled **********")
        logger.info(" Num examples = %d", len(examples))

    else:
        raise ValueError("(evaluate and dataname) parameters error !")

    if bert_name == 'roberta':
        tokenizer = args.roberta_tokenizer
    if bert_name == 'bert':
        tokenizer = args.bert_tokenizer

    features = data_utils.convert_examples_to_features(examples, args.max_seq_length, tokenizer, bert_name=bert_name, printN=printN,)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    if base_feature:
        return dataset

    if not base_feature:
        logger.info("............. Predict ing ...............")
        _, pseudo_labels = pseudo_label_to_unlabeled_data(args=args, dataname="unlabeled", model=trained_model, dataset=dataset)
        logger.info("............. Predict Done ...............")

        features = data_utils.space_based_convert_examples_to_features(
            examples=examples, 
            max_seq_length=args.max_seq_length,
            pseudo_labels=pseudo_labels,
            tokenizer=tokenizer,
            bert_name='bert', 
            printN=printN,
            )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_pseudo_labels = torch.tensor([f.pseudo_label for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_pseudo_labels)
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
        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                    help="Rul evaluation during training at each logging step.")
    parser.add_argument('--eval_logging_steps', type=int, default=100, help="Log every X evalution steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")

    parser.add_argument("--max_seq_length", default=512, type=int)
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
    parser.add_argument("--unlabeled_num", default=4000, type=int, help="how much labeled sample")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    Instance = SATokenizer(bert_name='bert_base', archive='./models/pre-trained_model/bert_uncased_L-12_H-768_A-12')
    args.bert_tokenizer = Instance.tokenizer

    trained_model = BertForClassification.from_pretrained(args.output_dir, num_labels=2)
    trained_model.cuda()
    trained_model.eval()

    train_dataset = load_and_cache_examples(args=args, trained_model=None, base_feature=True, bert_name='bert', dataname="train")
    unlabeled_dataset = load_and_cache_examples(args=args, trained_model=trained_model, base_feature=False, bert_name='bert', dataname="unlabeled")
    dev_dataset = load_and_cache_examples(args=args, trained_model=None, base_feature=True, bert_name='bert', dataname="dev")
    test_dataset = load_and_cache_examples(args=args, trained_model=None, base_feature=True, bert_name='bert', dataname="test")
    del trained_model

    model = BertaForClassificationModel.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], num_labels=2)
    domainM = BertForDomain.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    domainM.cuda()
    model.cuda()
    model.domain.load_state_dict(domainM.state_dict())

    if args.do_train:
        train(args=args, model=model, train_dataset=train_dataset, pseudo_labeled_dataset=unlabeled_dataset, eval_dataset=dev_dataset)
    if args.do_eval:
        evaluate(args=args, model=model, eval_dataset=train_dataset, dataname="test", domain='source')
        evaluate(args=args, model=model, eval_dataset=test_dataset, dataname="test", domain='target')


if __name__ == "__main__":
    main()
