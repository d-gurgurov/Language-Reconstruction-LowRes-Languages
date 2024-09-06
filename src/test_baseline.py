import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_metric, load_dataset
import torch # type: ignore
import sys


# loading the trained model and tokenizer
model_dir = '/netscratch/dgurgurov/projects2024/mt_lrls/models/swahili/safecheck'
# model_dir = 'Helsinki-NLP/opus-mt-en-sw'

model = MarianMTModel.from_pretrained(model_dir).to('cuda') # type: ignore 
tokenizer = MarianTokenizer.from_pretrained(model_dir)

# reading the Tatoeba test set
tatoeba_test_path = '/netscratch/dgurgurov/projects2024/mt_lrls/data/test_swahili/test_tatoeba_04_12.txt'
tatoeba_test_data = pd.read_csv(tatoeba_test_path, sep='\t', header=None, names=['eng', 'swh', 'eng_sent', 'swh_sent'])

# tokenizing and predicting
def translate_sentences(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        translated_tokens = model.generate(**inputs) # type: ignore
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translations

# getting predictions
predictions = translate_sentences(tatoeba_test_data['eng_sent'].tolist())

bleu_metric = load_metric('sacrebleu')

decoded_preds = [pred.split() for pred in predictions]
decoded_labels = [[label.split()] for label in tatoeba_test_data['swh_sent'].tolist()]

bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels) # type: ignore
print(f"BLEU score: {bleu_score}")

# saving the results
output_df = pd.DataFrame({
    'eng_sent': tatoeba_test_data['eng_sent'],
    'swh_sent': tatoeba_test_data['swh_sent'],
    'predicted_swh_sent': predictions
})

# output_df.to_csv('/netscratch/dgurgurov/thesis/mt_lrls/tatoeba_test_results.csv', index=False)

print("Translation and evaluation complete. Results saved to tatoeba_test_results.csv.")
