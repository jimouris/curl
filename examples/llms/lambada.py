#!/usr/bin/env python3

"""
python examples/llms/lambada.py --world_size 2 --model BertBase
"""

import argparse
import codecs
import logging
import os
import torch
import time

from datasets import load_dataset, load_from_disk
from math import ceil, log2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

import curl
import curl.communicator as comm
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher
from examples.llms.gpt2 import GPT2


def load_tsv(data_file):
    '''Load a tsv '''
    sentences = []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for row in data_fh:
            row = row.strip()
            row = eval(row)
            sentences.append(row)
    return sentences

def get_gpt_model(path, encyrpted_model):
    model = GPT2LMHeadModel.from_pretrained(path)
    model.eval()

    curl_model = encyrpted_model()
    curl_model.load_state_dict(model.state_dict())

    # Increase the vocabulary size to the next power of two.
    # This is used for correctness in the 'evaluate_embed' function.
    weight = curl_model.transformer.wte.weight
    new_size = pow(2, ceil(log2(weight.size()[0]))) - weight.size()[0]
    append = torch.zeros(new_size, weight.size()[1])
    curl_model.transformer.wte.weight = torch.cat((weight, append))

    curl_model.encrypt(src=0)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    return curl_model, tokenizer, model

def get_predictions(tokenizer, predictions):
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                 'very', 'having', 'with', 'they', 'own', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'a', 'an',
                 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
                 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                 'same', 'and', 'been', 'have', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'in',
                 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                 'if', 'theirs', 'my', 'against',  'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than',
                 ',', '.', '...', '?', '!', "'", "''", "", '?"', "?'", ',"', '."', "'s", ':', '"', '-', '�', '—'}

    _, predicted_token_ids = torch.topk(predictions[0, -1, :], k=128)
    for candidate in predicted_token_ids:
        candidate = tokenizer.decode([candidate]).strip()
        if candidate.lower() not in stopwords or candidate.lower() == target_word.lower()[:len(candidate)]:
            predicted_word = candidate
            break
    assert predicted_word is not None, "No candidate word found"
    return predicted_word

def evaluate_gpt2_on_lambada(tokenizer, model, curl_model, data="tsv"):
    # Load pre-trained GPT-2 tokenizer and model
    # Load the LAMBADA dataset
    if data == "cimec":
        dataset = load_dataset('cimec/lambada', split='test')
    elif data == "tsv":
        dataset = load_tsv('examples/llms/glue_data/lambada_test.jsonl')
    else:
        dataset = load_from_disk('examples/llms/glue_data/lambada_test.jsonl')
    print('LAMBADA loaded')

    correct_predictions = 0
    curl_correct_predictions = 0
    total_predictions = 0

    for example in tqdm(dataset):
        total_predictions += 1
        text = example['text']
        # Split the text into context and target (last word)
        *context, target_word = text.split()
        context = ' '.join(context)

        # Tokenize context
        inputs = tokenizer(context, return_tensors='pt')
        input_ids = inputs['input_ids']

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits

        curl_outputs = curl_model(curl.cryptensor(input_ids, precision=0))
        curl_predictions = curl_outputs.get_plain_text()

        # Get the predicted token
        next_word = get_predictions(tokenizer, predictions)
        with torch.no_grad():
            predicted_word = ""
            context += ' '
            for i in range(10):
                predicted_word += next_word
                context += next_word
                if predicted_word.lower() == target_word.lower() or predicted_word.lower() != target_word.lower()[:len(predicted_word)]:
                    break
                input_ids = tokenizer(context, return_tensors='pt')['input_ids']
                outputs = model(input_ids)
                predictions = outputs.logits
                next_word = tokenizer.decode([torch.argmax(predictions[0, -1, :])]).strip()

        # Compare with the actual last word
        if predicted_word.lower() == target_word.lower():
            correct_predictions += 1

        next_word = get_predictions(tokenizer, curl_predictions)
        predicted_word = ""
        context += ' '
        for i in range(10):
            predicted_word += next_word
            context += next_word
            if predicted_word.lower() == target_word.lower() or predicted_word.lower() != target_word.lower()[:len(predicted_word)]:
                break
            input_ids = tokenizer(context, return_tensors='pt')['input_ids']
            outputs = curl_model(curl.cryptensor(input_ids, precision=0))
            predictions = outputs.get_plain_text()
            next_word = tokenizer.decode([torch.argmax(predictions[0, -1, :])]).strip()

        if predicted_word.lower() == target_word.lower():
            curl_correct_predictions += 1

        print(f'LAMBADA Torch Accuracy: {correct_predictions / total_predictions:.4f}')
        print(f'LAMBADA Curl  Accuracy: {curl_correct_predictions / total_predictions:.4f}')

    accuracy = correct_predictions / total_predictions
    print(f'LAMBADA Accuracy: {accuracy:.4f}')
    return accuracy, curl_accuracy


def run_lambada(cfg_file, model, count=100, communication=False, device=None):
    # First cold run.
    curl.init(cfg_file, device=device)
    if communication:
        comm.get().set_verbosity(True)

    if model == "GPT2":
        curl_model, tokenizer, model = get_gpt_model("gpt2", GPT2)
        print('GPT2 loaded')

    base_accuracy, curl_accuracy = evaluate_gpt2_on_lambada(tokenizer, model, curl_model)
    logging.info(f"Base Accuracy: {base_accuracy}")
    logging.info(f"Curl Accuracy: {curl_accuracy}")

    if communication:
        comm.get().print_communication_stats()
        exit(0)


def get_args():
    parser = argparse.ArgumentParser(description="Curl LLM LAMBADA Test")
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="The number of parties to launch. Each party acts as its own process",
    )
    parser.add_argument(
        "--evaluator_size",
        "-es",
        type=int,
        default=0,
        help="The number of eval parties to launch. Each party acts as its own process",
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
    models = ['GPT2']
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
    elif args.evaluator_size:
        logging.info("Using Fission config")
        cfg_file = cfg_file.replace("default", "fission")
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
    run_lambada(cfg_file, args.model, args.count, args.communication, args.device)

    print('Done')

def main():
    args = get_args()
    cfg_file = get_config(args)
    curl.cfg.load_config(cfg_file)

    if args.communication and cfg.mpc.provider == "TTP":
        raise ValueError("Communication statistics are not available for TTP provider")

    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, args.evaluator_size, _run_experiment, args, cfg_file)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        _run_experiment(args)


if __name__ == "__main__":
    main()
