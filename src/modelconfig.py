MODEL_ARCHIVE_MAP = {
    'bert_base':  './pre-trained_model/bert_uncased_L-12_H-768_A-12',
    'roberta_base':  './pre-trained_model/roberta_base',
    'bert_large':  './pre-trained_model/bert_large_wwm_uncased_L-24_H-1024_A-16',
    'bert_extend': './pre-trained_model/post_trained_model_review/pt_bert-base-uncased_amazon_yelp',
    'bert_laptop': './pre-trained_model/post_trained_model_review/laptop_pt_review',  # The BERT-base-uncased model fine-tuend in a in-domain corpus
    'bert_rest': './pre-trained_model/post_trained_model_review/rest_pt_review',  # The BERT-base-uncased model fine-tuend in a in-domain corpus
}
