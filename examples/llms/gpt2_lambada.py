#!/usr/bin/env python3

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import logging
import curl
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from curl.config import cfg
import curl.communicator as comm


def evaluate_clear_gpt2_on_lambada():
    # Load the LAMBADA dataset
    dataset = load_dataset('cimec/lambada', split='test')
    print('LAMBADA loaded')

    # Load pre-trained GPT-2 tokenizer and model
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    print('GPT2 loaded')

    correct_predictions = 0
    total_predictions = 0
    for example in dataset:
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

        # Get the predicted token
        predicted_token_id = torch.argmax(predictions[0, -1, :]).item()
        predicted_word = tokenizer.decode([predicted_token_id]).strip()

        # Compare with the actual last word
        if predicted_word == target_word:
            correct_predictions += 1
        total_predictions += 1
        if total_predictions >= 10:
            break
    accuracy = correct_predictions / total_predictions
    print(f'LAMBADA Accuracy [Dataset size: {total_predictions}]: {accuracy:.4f}')


# def evaluate_private_gpt2_on_lambada(dataset, model, tokenizer):
#     correct_predictions = 0
#     total_predictions = 0
#     for example in dataset:
#
#
#         if predicted_word == target_word:
#             correct_predictions += 1
#         total_predictions += 1
#         if total_predictions >= 10:
#             break
#     accuracy = correct_predictions / total_predictions
#     print(f'LAMBADA Accuracy [Dataset size: {total_predictions}]: {accuracy:.4f}')


def run_gpt2_lambada(cfg_file, fill_cache=False, communication=False, device=None):
    curl.init(cfg_file, device=device)
    if communication:
        comm.get().set_verbosity(True)

    functions_data = cfg.config.get('functions', {})
    filtered_data = {
        key: dict(value)['method'] for key, value in functions_data.items() if 'method' in dict(value)
    }
    logging.info("\t'{}'".format(filtered_data))

    provider = curl.mpc.get_default_provider()
    if fill_cache:
        logging.info(f"=" * 22 + " Tracing requests for the cache " + "=" * 22)
        provider.trace_once()
    else:
        provider.load_cache()

    # Load pre-trained GPT-2 tokenizer and model
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Obtain the model by running: python3 -m transformers.onnx --model=gpt2 onnx/ --opset=20
    onnx_file_path = "./examples/llms/model.onnx"
    print(f'Loading GPT2 with ONNX: {onnx_file_path}')
    with open(onnx_file_path, "rb") as f:
        private_model = curl.nn.from_onnx(f, track_execution=True)
        print(type(private_model))
        print('Encrypting the model')
        private_model.encrypt()
        print(type(private_model))
    print('Successfully loaded encrypted GPT2')

    print('Tokenizing')
    my_input = "What is your name?"
    tokenized_input = tokenizer(my_input, return_tensors='pt')
    print(f"{tokenized_input=}")
    tokenized_input_ids_enc = curl.cryptensor(tokenized_input['input_ids'], precision=0)
    print(f'{tokenized_input['input_ids']=}')
    print(f'{tokenized_input_ids_enc.get_plain_text()=}')

# tokenized_input_ids_enc.get_plain_text()=tensor([[2061,  318,  534, 1438,   30]])
#  x: array([[0.031448, 0.004852, 0.008148, 0.021942, 0.000458]], dtype=float32)
#  y: array([[2061,  318,  534, 1438,   30]])

    tokenized_attention_mask_enc = curl.cryptensor(tokenized_input['attention_mask'])
    print('running private model')
    private_output = private_model(tokenized_input_ids_enc, tokenized_attention_mask_enc)
    print(f'{private_output=}')

    logging.info("="*60)

    if fill_cache:
        logging.info(f"=" * 22 + " Filling the cache " + "=" * 22)
        provider.fill_cache()

    if communication:
        comm.get().print_communication_stats()
        exit(0)
    print('Done')
