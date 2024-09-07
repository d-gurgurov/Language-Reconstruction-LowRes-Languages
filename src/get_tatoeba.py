import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_metric, load_dataset
import torch # type: ignore
import argparse
from tqdm import tqdm


# parsing command line arguments
parser = argparse.ArgumentParser(description='Testing a model on Tatoeba')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "swh" for Swahili)')
parser.add_argument('--model', type=str, required=True, help='Model configuration')
parser.add_argument('--helsinki', type=bool, required=False, default=False, help='Use Helsinki-NLP model?')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
model_type = args.model
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}


# loading the trained model and tokenizer
model_dir = f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/{model_type}'

if args.model:
    model_dir = f'Helsinki-NLP/opus-mt-en-{language}'

model = MarianMTModel.from_pretrained(model_dir).to('cuda') # type: ignore 
tokenizer = MarianTokenizer.from_pretrained(model_dir)

# reading the Tatoeba test set
tatoeba_test_path = f'/netscratch/dgurgurov/projects2024/mt_lrls/data/test_{lang_map[language]}/test_tatoeba_{language}.txt'
tatoeba_test_data = pd.read_csv(tatoeba_test_path, sep='\t', header=None, names=['eng', f'{language}', 'eng_sent', f'{language}_sent'])

print(f"Length of the Tatoeba test dataset for {lang_map[language]}:", len(tatoeba_test_data))

# tokenizing and predicting
def translate_sentences_in_batches(sentences, batch_size=64):
    translations = []
    # processing sentences in batches
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)  # type: ignore
        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)  # adding batch translations to the full list
    return translations

# getting predictions
predictions = translate_sentences_in_batches(tatoeba_test_data['eng_sent'].tolist())

bleu_metric = load_metric('sacrebleu')

decoded_preds = [pred.split() for pred in predictions]
decoded_labels = [[label.split()] for label in tatoeba_test_data[f'{language}_sent'].tolist()]

bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels) # type: ignore
print(f"BLEU score: {bleu_score}")

# saving the results
output_df = pd.DataFrame({
    'eng_sent': tatoeba_test_data['eng_sent'],
    f'{language}_sent': tatoeba_test_data[f'{language}_sent'],
    'predicted_swh_sent': predictions
})
