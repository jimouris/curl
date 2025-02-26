#!/usr/bin/env python3

"""
python examples/llms/mmlu.py --world_size 2 --model SmolLM-135m
"""

import argparse
import codecs
import logging
import os
import torch
from math import ceil, log2
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

import curl
import curl.communicator as comm
from curl.config import cfg
from examples.multiprocess_launcher import MultiProcessLauncher
from examples.llms.models.bert_for_sequence_classification import BertBaseForSequenceClassification, BertTinyForSequenceClassification

from datasets import load_dataset
from tqdm import tqdm


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

    # Increase the vocabulary size to the next power of two.
    # This is used for correctness in the 'evaluate_embed' function.
    weight = curl_bert_model.bert.embeddings.word_embeddings.weight
    new_size = pow(2, ceil(log2(weight.size()[0]))) - weight.size()[0]
    append = torch.zeros(new_size, weight.size()[1])
    curl_bert_model.bert.embeddings.word_embeddings.weight = torch.cat((weight, append))

    curl_bert_model.encrypt(src=0)
    bert_tokenizer = AutoTokenizer.from_pretrained(path)
    return curl_bert_model, bert_tokenizer, bert_model


def run_qnli_accuracy_test(model, curl_model, data, targets, total):
    count = 312
    count_enc = 299
    print(f"{total=}")
    start = 344
    now = time.time()
    for label in range(start, total):
        # Plaintext
        outputs = model(**data[label])
        result = outputs.logits
        count += targets[label] == result.argmax()
        # Encrypted
        x_enc = {}
        x_enc['input_ids'] = curl.cryptensor(data[label]["input_ids"], precision = 0)
        x_enc['token_type_ids'] = curl.cryptensor(data[label]["token_type_ids"], precision = 0)
        outputs_enc = curl_model(**x_enc)
        result_enc = outputs_enc.get_plain_text()
        print(f"{result=}, {result_enc=}")
        count_enc += targets[label] == result_enc.argmax()
        print(f"{label=}, time={time.time()-now}, {count=}, {count_enc=}, {count/label=}, {count_enc/label=}")
    return count / total, count_enc / total


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

    base_accuracy, curl_accuracy = run_qnli_accuracy_test(bert_model, curl_bert_model, data, targets, count)
    logging.info(f"Base Accuracy: {base_accuracy}")
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
    run_qnli(cfg_file, args.model, args.count, args.communication, args.device)

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
    # main()

    # Load SmolLM model and tokenizer
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"  # Replace with actual model name on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load MMLU dataset
    dataset = load_dataset("NLPCoreTeam/mmlu_ru", "abstract_algebra")

    # Evaluation function
    correct = 0
    total = 0

    for example in tqdm(dataset['test']):
        question = example['question_en']
        choices = example['choices_en']
        answer = example['answer']

        # Prepare input prompt
        prompt = f"{question}\nChoices: {', '.join(choices)}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate model output
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1
        )

        # Decode and process output
        print(f"{outputs=}")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{generated_text=}")
        print(f"{generated_text.split('Answer:')[-1]=}")
        try:
            print(f"{generated_text.split('Answer:')[-1].strip().split()[0]=}")
            predicted_answer = generated_text.split("Answer:")[-1].strip().split()[0]
        except Exception as e:
            print(generated_text.split('Answer:')[-1].strip().split())
            predicted_answer = "None"

        # Compare predicted answer with ground truth
        if predicted_answer == answer:
            correct += 1
        total += 1

    accuracy = correct / total

    # Run evaluation
    print(f"Accuracy on MMLU: {accuracy * 100:.2f}%")
