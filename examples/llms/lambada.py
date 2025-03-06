#!/usr/bin/env python3

"""
python examples/llms/lambada.py --multiprocess
"""

import argparse
import codecs
import logging
import os
import time
import torch

from datasets import load_dataset, load_from_disk
from math import ceil, log2
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
from tqdm import tqdm

import curl
import curl.communicator as comm
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher


def load_tsv(data_file):
    '''Load a tsv '''
    sentences = []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for row in data_fh:
            row = row.strip()
            row = eval(row)
            sentences.append(row)
    return sentences

def load_data(mode):
    # Load the LAMBADA dataset
    match mode:
        case "cimec":
            dataset = load_dataset('cimec/lambada', split='test')
        case "tsv":
            dataset = load_tsv('examples/llms/glue_data/lambada_test.jsonl')
        case "disk":
            dataset = load_from_disk('examples/llms/glue_data/lambada_test.jsonl')
        case _:
            raise ValueError(f"Invalid data mode {mode}")
    return dataset

def get_gpt_model(path, mode, device):
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    if mode in ("Neo", "NeoSecret"):
        model = GPTNeoForCausalLM.from_pretrained(path)
    else:
        model = GPT2LMHeadModel.from_pretrained(path)
    model.eval()
    model.to(device)

    match mode:
        case "Clear":
            from examples.llms.models.gpt_clear import GPT2LMHead as GPTLMHead
        case "Fixed":
            from examples.llms.models.gpt_fixed import GPT2LMHead as GPTLMHead
        case "Secret":
            from examples.llms.models.gpt_curl import GPT2LMHead as GPTLMHead
        case "Neo":
            from examples.llms.models.gpt_neo import GPTNeoLMHead as GPTLMHead
        case "NeoSecret":
            from examples.llms.models.gpt_neo_curl import GPTNeoLMHead as GPTLMHead
        case _:
            raise ValueError(f"Invalid model mode {mode}")

    curl_model = GPTLMHead()
    curl_model.load_state_dict(model.state_dict())
    curl_model.to(device)

    if mode in ("Secret", "NeoSecret"):
        # Increase the vocabulary size to the next power of two.
        # This is used for correctness in the 'evaluate_embed' function.
        weight = curl_model.transformer.wte.weight
        new_size = pow(2, ceil(log2(weight.size()[0]))) - weight.size()[0]
        append = torch.zeros(new_size, weight.size()[1])
        curl_model.transformer.wte.weight = torch.cat((weight, append))
        curl_model.encrypt(src=0)
    return tokenizer, model, curl_model

def get_predictions(tokenizer, predictions, target_word):
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

    # Get the predicted token
    _, predicted_token_ids = torch.topk(predictions[0, -1, :], k=128)
    predicted_word = None
    for candidate in predicted_token_ids:
        candidate = tokenizer.decode([candidate]).strip()
        if candidate.lower() not in stopwords:
            predicted_word = candidate
            break
    assert predicted_word is not None, "No candidate word found"

    # Compare with the actual last word
    return predicted_word.lower() == target_word.lower()

def evaluate_lambada(mode, data="tsv", device=torch.device("cpu"), secret=False):
    path = "EleutherAI/gpt-neo-1.3B" if mode in ("Neo", "NeoSecret") else "gpt2"
    tokenizer, model, curl_model = get_gpt_model(path, mode, device)
    dataset = load_data(data)

    correct_predictions = 0
    curl_correct_predictions = 0
    total_predictions = 0
    now = time.time()

    print("Starting")
    for example in tqdm(dataset):
        total_predictions += 1

        # Split the text into context and target (last word)
        text = example['text']
        *context, target_word = text.split()
        context = ' '.join(context)

        # Tokenize context
        inputs = tokenizer(context, return_tensors='pt')
        input_ids = inputs['input_ids']

        # Get model predictions
        if not secret:
            with torch.no_grad():
                outputs = model(input_ids.to(device))
                predictions = outputs.logits
            correct_predictions += get_predictions(tokenizer, predictions, target_word)

        if mode in ("Secret", "NeoSecret"):
            curl_outputs = curl_model(curl.cryptensor(input_ids, device=device, precision=0))
            curl_predictions = curl_outputs.get_plain_text()
        else:
            curl_predictions = curl_model(input_ids.to(device))
        curl_correct_predictions += get_predictions(tokenizer, curl_predictions, target_word)

        print(f'LAMBADA Torch Accuracy: {correct_predictions / total_predictions:.4f} ({correct_predictions})')
        print(f'LAMBADA Curl  Accuracy: {curl_correct_predictions / total_predictions:.4f} ({curl_correct_predictions})')
        print(f'time={time.time()-now}')

    accuracy = correct_predictions / total_predictions
    curl_accuracy = curl_correct_predictions / total_predictions
    return accuracy, curl_accuracy


def run_lambada(cfg_file, communication=False, device=None, mode="Clear", data="tsv", secret=False):
    # First cold run.
    if mode in ("Secret", "NeoSecret"):
        curl.init(cfg_file, device=device)
        if communication:
            comm.get().set_verbosity(True)

    base_accuracy, curl_accuracy = evaluate_lambada(mode, data, device, secret)

    logging.info(f"Base Accuracy: {base_accuracy}")
    logging.info(f"Curl Accuracy: {curl_accuracy}")

    if mode in ("Secret", "NeoSecret") and communication:
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
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        default="cpu",
        help="The device to run the benchmarks",
    )
    parser.add_argument(
        "--multi-gpu",
        "-mg",
        required=False,
        default=False,
        action="store_true",
        help="Use different gpu for each party. Will override --device if selected",
    )
    models=["Clear", "Fixed", "Secret", "Neo", "NeoSecret"]
    parser.add_argument(
        "--model",
        choices=models,
        required=True,
        help="Choose a model to run from the following options: {}".format(models),
    )
    data=["cimec", "tsv", "disk"]
    parser.add_argument(
        "--data",
        choices=data,
        default="tsv",
        help="Choose a data format from the following options: {}".format(data),
    )
    parser.add_argument(
        "--secret",
        default=False,
        action="store_true",
        help="Do not run plaintext model",
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
    run_lambada(cfg_file, args.communication, args.device, args.model, args.data, args.secret)

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
