import os
import sys
sys.path.append('../')
sys.path.append('../../')
import re
import logging
import random
from collections import defaultdict, namedtuple

import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


class SATokenizer(object):
    def __init__(self, bert_name='bert_base', archive=None):
        if bert_name in ['bert_base', 'bert_large']:
            self.tokenizer = BertTokenizer.from_pretrained(archive)
            self.name = 'bert'
        elif bert_name in ['roberta_base', 'roberta_large']:
            self.tokenizer = RobertaTokenizer.from_pretrained(archive, tokenizer_class="RobertaTokenizer")
            self.name = 'roberta'
        else:
            raise ValueError

    def convert_tag_to_ids(self, tags, tag_vocab_dict):
        new_tags = []
        for t in tags:
            if t in tag_vocab_dict.keys():
                new_tags.append(t)
            else:
                new_tags.append('[UNK]')
        return [tag_vocab_dict[t] for t in new_tags]


def XML2arrayRAW(path):
    logger.info('load data from {}'.format(os.path.join(path)))
    reviews = []
    tree = ET.parse(path)
    root = tree.getroot()
    for rev in root.iter('review'):
        reviews.append(rev.text)
    return reviews


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, tag=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.tag = tag


class InputFeatures(object):
    def __init__(self, 
        input_ids, 
        input_mask, 
        clue_ids=None,
        senti_type=None,
        segment_ids=None,
        masked_indice=None, 
        masked_lm_label=None, 
        label_id=None,
        pseudo_label=None,
        tag_id=None,
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.clue_ids = clue_ids
        self.senti_type = senti_type
        self.segment_ids = segment_ids
        self.masked_indice = masked_indice
        self.masked_lm_label = masked_lm_label
        self.label_id = label_id
        self.pseudo_label = pseudo_label
        self.tag_id = tag_id


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    def insertspace(self, text):
        text = re.sub( r'([a-zA-Z])([,.!])', r'\1 \2', text)
        text = re.sub( r'([,.!])([a-zA-Z])', r'\1 \2', text)
        return text
    def read_txt(self, data_dir, key, _type):
        lines = {}
        id = 0
        posReviews = XML2arrayRAW(data_dir +'/' + key + '/positive.parsed')
        negReviews = XML2arrayRAW(data_dir +'/' + key + '/negative.parsed')
        indexs = random.sample(range(0, 1000), 200)        
        if _type == "train":
            posReviews = [text for i, text in enumerate(posReviews) if i not in indexs]
            negReviews = [text for i, text in enumerate(negReviews) if i not in indexs]
        elif _type == "dev":
            posReviews = [text for i, text in enumerate(posReviews) if i in indexs]
            negReviews = [text for i, text in enumerate(negReviews) if i in indexs]
        elif _type == "test":
            pass

        for review in posReviews:
            review = self.insertspace(review.replace('\n', ' '))
            lines[id] = {'sentence': review, 'label': 'positive'}
            id += 1

        for review in negReviews:
            review = self.insertspace(review.replace('\n', ' '))
            lines[id] = {'sentence': review, 'label': 'negative'}
            id += 1
        return lines
    def read_untxt(self, data_dir, key, _type):
        lines = {}
        id = 0
        reviews = XML2arrayRAW(data_dir +'/' + key + '/{}UN.txt'.format(key))
        for review in reviews:
            review = self.insertspace(review.replace('\n', ' '))
            lines[id] = {'sentence': review, 'label': 'positive'} # To simplify, do not add new labels
            id += 1
        return lines


class SAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        data_dir, domain = data_dir.rsplit('/', 1)
        source, _ = domain.split('-')
        return self._create_examples(self.read_txt(data_dir, source, "train"), set_type="train")

    def get_dev_examples(self, data_dir):
        data_dir, domain = data_dir.rsplit('/', 1)
        source, _ = domain.split('-')
        return self._create_examples(self.read_txt(data_dir, source, "dev"), set_type="dev")

    def get_test_examples(self, data_dir):
        data_dir, domain = data_dir.rsplit('/', 1)
        _, target = domain.split('-')
        return self._create_examples(self.read_txt(data_dir, target, "test"), set_type="test")

    def get_target_unexamples(self, data_dir):
        data_dir, domain = data_dir.rsplit('/', 1)
        _, target = domain.split('-')
        return self._create_examples(self.read_untxt(data_dir, target, "unlabel"), set_type="unlabel")

    def get_labels(self):
        return ['negative', 'positive']

    def _create_examples(self, lines, set_type):
        examples = []
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, i)
            text_a = lines[i]['sentence']
            label = lines[i]['label']
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(
    examples, 
    max_seq_length, 
    tokenizer, 
    bert_name='bert', 
    printN=1,
    ):
    label_map = {'negative':0, 'positive':1}
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = []
        if bert_name == 'bert':      
            tokens_a = tokenizer.tokenize(example.text_a.lower())
        if bert_name == 'roberta':
            tokens_a = tokenizer._tokenize(example.text_a.lower())

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = [cls_token] + tokens_a + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token_id)
            input_mask.append(0)
        assert len(input_ids) == len(input_mask) == max_seq_length
        label_id = label_map[example.label]

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, label_id=label_id,))

        if ex_index < printN:
            logger.info("*** Source or Target Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("source / target tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("source / target input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("sentiment: %s" % example.label)
            logger.info("label: %d" % label_id)
    return features


def clue_based_convert_examples_to_features(
    examples=None, 
    labeled_examples=None, ## rouge & bleu
    max_seq_length=None, 
    retrieval_num=None,
    indices=None,
    clue_labels=None,
    pseudo_labels=None,
    tokenizer=None,
    printN=1):
    label_map = {'negative':0, 'positive':1}
    pad_token_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
    assert len(examples) == indices.size(0) == clue_labels.size(0) == pseudo_labels.size(0)

    features = []
    for ex_index, (example, indice, clue_label, pseudo_label) in enumerate(zip(examples, indices.tolist(), clue_labels.tolist(), pseudo_labels.tolist())):
        tokens_a = tokenizer.tokenize(example.text_a.lower())

        # Account for [CLS] xxxx [SEP] ------------ [S1]  [S2] .... [SN]  [SEP] 
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token_id)
            input_mask.append(0)
        assert len(input_ids) == len(input_mask) == max_seq_length

        label_id = label_map[example.label]
        features.append(InputFeatures(
            input_ids=input_ids, 
            input_mask=input_mask,
            clue_ids=indice,
            senti_type=clue_label, 
            label_id=label_id,
            pseudo_label=[round(pl, 3) for pl in pseudo_label], 
            ))

        if ex_index < printN:
            logger.info("*** Source or Target Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("source / target tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("source / target input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("clue ids: %s" % " ".join([str(x) for x in indice]))
            logger.info("senti type: %s" % " ".join([str(x) for x in clue_label]))
            logger.info("pesudo label: %s" % " ".join([str(x) for x in pseudo_label]))
    return features


def space_based_convert_examples_to_features(examples=None, max_seq_length=None, pseudo_labels=None, tokenizer=None, bert_name='bert', printN=1,):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
    assert len(examples) == pseudo_labels.size(0)

    features = []
    for ex_index, (example, pseudo_label) in enumerate(zip(examples, pseudo_labels.tolist())):
        tokens_a = []
        if bert_name == 'bert':      
            tokens_a = tokenizer.tokenize(example.text_a.lower())
        if bert_name == 'roberta':
            tokens_a = tokenizer._tokenize(example.text_a.lower())

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = [cls_token] + tokens_a + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token_id)
            input_mask.append(0)
        assert len(input_ids) == len(input_mask) == max_seq_length

        features.append(InputFeatures(
            input_ids=input_ids, 
            input_mask=input_mask,
            pseudo_label=[round(pl, 3) for pl in pseudo_label], 
            ))

        if ex_index < printN:
            logger.info("*** Source or Target Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info(" target tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info(" target input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(" pesudo label: %s" % " ".join([str(x) for x in pseudo_label]))
    return features

