#!/usr/bin/env python3

"""
python examples/llms/qnli.py --world_size 2 --model BertBase
"""

import argparse
import codecs
import logging
import os
from transformers import AutoTokenizer, BertForSequenceClassification

import curl
import curl.communicator as comm
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher
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
        outputs = model(**x)
        if encrypted:
            outputs = outputs.get_plain_text()
        else:
            outputs = outputs.logits
        count += targets[label] == outputs.argmax()
    return count / total


def run_qnli(cfg_file, model, count=100, communication=False, device=None):
    # First cold run.
    curl.init(cfg_file, device=device)
    if communication:
        comm.get().set_verbosity(True)

    if model == "BertBase":
        curl_bert_model, bert_tokenizer, bert_model = get_bert_model("gchhablani/bert-base-cased-finetuned-qnli", BertBaseForSequenceClassification)
    elif model == "BertTiny":
        curl_bert_model, bert_tokenizer, bert_model = get_bert_model("M-FAC/bert-tiny-finetuned-qnli", BertTinyForSequenceClassification)

    data, targets = load_tsv("examples/llms/glue_data/QNLI/dev.tsv", bert_tokenizer)

    if count < 1:
        count = len(data)

    base_accuracy = run_qnli_accuracy_test(bert_model, data, targets, count, encrypted=False)
    logging.info(f"Base Accuracy: {base_accuracy}")
    curl_accuracy = run_qnli_accuracy_test(curl_bert_model, data, targets, count, encrypted=True)
    logging.info(f"Curl Accuracy: {curl_accuracy}")

    if communication:
        comm.get().print_communication_stats()
        exit(0)


def get_args():
    parser = argparse.ArgumentParser(description="Curl LLM QNLI Test")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--multiprocess",
        default=False,
        action="store_true",
        help="Run example in multiprocess mode",
    )
    parser.add_argument(
        "--approximations",
        default=False,
        action="store_true",
        help="Use approximations for non-linear functions",
    )
    parser.add_argument(
        "--no-cmp",
        default=False,
        action="store_true",
        help="Use LUTs for bounded functions without comparisons",
    )
    parser.add_argument(
        "--communication",
        default=False,
        action="store_true",
        help="Print communication statistics",
    )
    models = ['BertTiny', 'BertBase']
    parser.add_argument(
        "--model",
        choices=models,
        required=True,
        help="Choose a model to run from the following options: {}".format(models),
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=-1,
        help="The number of samples to iterate over. -1 for entire dataset",
    )
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        default="cpu",
        help="the device to run the benchmarks",
    )
    parser.add_argument(
        "--multi-gpu",
        "-mg",
        required=False,
        default=False,
        action="store_true",
        help="use different gpu for each party. Will override --device if selected",
    )
    args = parser.parse_args()
    return args

def get_config(args):
    cfg_file = curl.cfg.get_default_config_path()
    if args.approximations:
        logging.info("Using Approximation Config:")
        cfg_file = cfg_file.replace("default", "approximations")
    elif args.no_cmp:
        logging.info("Using config with LUTs without comparisons:")
        cfg_file = cfg_file.replace("default", "llm_config")
    else:
        logging.info("Using LUTs Config:")
    return cfg_file

def _run_experiment(args):
    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)

    cfg_file = get_config(args)
    run_qnli(cfg_file, args.model, args.count, args.communication, args.device)

    print('Done')

def main():
    args = get_args()
    cfg_file = get_config(args)
    curl.cfg.load_config(cfg_file)

    if args.communication and cfg.mpc.provider == "TTP":
        raise ValueError("Communication statistics are not available for TTP provider")

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, _run_experiment, args, cfg_file)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)


if __name__ == "__main__":
    main()
