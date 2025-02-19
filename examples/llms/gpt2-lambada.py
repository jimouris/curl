from string import punctuation

import codecs
import torch

from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm


def load_tsv(data_file):
    '''Load a tsv '''
    sentences = []
    with codecs.open(data_file, 'r', 'utf-8') as data_fh:
        for row in data_fh:
            row = row.strip()
            row = eval(row)
            sentences.append(row)
    return sentences

def evaluate_gpt2_on_lambada(model_name='gpt2', data='tsv'):
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                 'very', 'having', 'with', 'they', 'own', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
                 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                 'same', 'and', 'been', 'have', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                 'if', 'theirs', 'my', 'against',  'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than',
                 ',', '.', '...', '?', '!', "'", "''", "", '?"', "?'", ',"', '."', "'s", ':', '"', '-', '�', '—', 'an', 'a', 'in'}

    model = GPT2LMHeadModel.from_pretrained(model_name)
    for name, param in model.named_parameters():
        print(f"{name} {param.shape}")
    model.eval()
    print('GPT2 loaded')

    # Load the LAMBADA dataset
    if data == "cimec":
        dataset = load_dataset('cimec/lambada', split='test')
    elif data == "tsv":
        dataset = load_tsv('examples/llms/glue_data/lambada_test.jsonl')
    else:
        dataset = load_from_disk('examples/llms/glue_data/lambada_test.jsonl')
    print('LAMBADA loaded')

    correct_predictions = 0
    total_predictions = 0

    for example in tqdm(dataset):
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
        _, predicted_token_ids = torch.topk(predictions[0, -1, :], k=128)
        for candidate in predicted_token_ids:
            candidate = tokenizer.decode([candidate]).strip()
            print(f"{candidate=}")
            if candidate.lower() not in stopwords:
                predicted_word = candidate
                break
        assert predicted_word is not None, "No candidate word found"

        with torch.no_grad():
            next_word = predicted_word
            predicted_word = ""
            for i in range(10):
                if predicted_word.lower() == target_word.lower():
                    break
                predicted_word += next_word
                context += ' ' + next_word
                input_ids = tokenizer(context, return_tensors='pt')['input_ids']
                outputs = model(input_ids)
                predictions = outputs.logits
                next_word = tokenizer.decode([torch.argmax(predictions[0, -1, :])]).strip()
                print(next_word)

        # predicted_word = tokenizer.decode([predicted_token_id]).strip()

        # Compare with the actual last word
        print(f'{target_word=}, {predicted_word=}')
        if predicted_word.lower() == target_word.lower():
            print('Correct')
            correct_predictions += 1
        else:
            print('Incorrect')
        total_predictions += 1
        # if total_predictions >= 10:
        #     break
        print(f'LAMBADA Accuracy: {correct_predictions / total_predictions:.4f}')

    accuracy = correct_predictions / total_predictions
    print(f'LAMBADA Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    evaluate_gpt2_on_lambada()
