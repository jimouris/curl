#!/usr/bin/env python3

import codecs
from transformers import AutoTokenizer, BertForSequenceClassification

import curl
from examples.llms.bert_for_sequence_classification import BertBaseForSequenceClassification, BertTinyForSequenceClassification


def load_tsv(data_file, tokenizer, delimiter='\t'):
    '''Load a tsv '''
    sentences = []
    targets = []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for _ in range(1):
            data_fh.readline()
        for row in data_fh:
            row = row.strip().split(delimiter)
            sentences.append(tokenizer(row[1][:512], row[2][:512], return_tensors="pt"))
            targets.append(1*(row[3] == "not_entailment"))
    return sentences, targets


def get_bert_model(path, encyrpted_model):
    bert_model = BertForSequenceClassification.from_pretrained(path)
    bert_model.eval()
    curl_bert_model = encyrpted_model()
    curl_bert_model.load_state_dict(bert_model.state_dict())
    curl_bert_model.encrypt(src=0)
    bert_tokenizer = AutoTokenizer.from_pretrained(path)
    return curl_bert_model, bert_tokenizer, bert_model


def run_qnli_accuracy_test(model, data, targets, total, encrypted=False):
    count = 0
    for label in range(total):
        if encrypted:
            x = {}
            x['input_ids'] = curl.cryptensor(data[label]["input_ids"], precision = 0)
            x['token_type_ids'] = curl.cryptensor(data[label]["token_type_ids"], precision = 0)
        else:
            x = data[label]
        outputs = model(x)
        if encrypted:
            outputs = outputs.get_plain_text()
        count += targets[label] == outputs.argmax()
    return count / total


def run_qnli(model, count=100):
    if model == "base":
        curl_bert_model, bert_tokenizer, bert_model = get_bert_model("gchhablani/bert-base-cased-finetuned-qnli", BertBaseForSequenceClassification)
    elif model == "tiny":
        curl_bert_model, bert_tokenizer, bert_model = get_bert_model("M-FAC/bert-tiny-finetuned-qnli", BertTinyForSequenceClassification)

    data, targets = load_tsv("GLUE-baselines/glue_data/QNLI/dev.tsv", bert_tokenizer)

    if count < 1:
        count = len(data)

    base_accuracy = run_qnli_accuracy_test(bert_model, data, targets, count, encrypted=False)
    curl_accuracy = run_qnli_accuracy_test(curl_bert_model, data, targets, count, encrypted=True)
    return base_accuracy, curl_accuracy