import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

def evaluate_gpt2_on_lambada(model_name='gpt2'):
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    for param in model.parameters():
        print(param.shape)
    model.eval()
    print('GPT2 loaded')

    # Load the LAMBADA dataset
    dataset = load_dataset('cimec/lambada', split='test')
    print('LAMBADA loaded')

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
        # if total_predictions >= 10:
        #     break

    accuracy = correct_predictions / total_predictions
    print(f'LAMBADA Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    evaluate_gpt2_on_lambada()
