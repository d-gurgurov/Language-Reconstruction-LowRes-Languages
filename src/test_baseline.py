import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_metric
import torch # type: ignore


# loading the trained model and tokenizer
# model_dir = '/netscratch/dgurgurov/thesis/mt_lrls/baseline/'
model_dir = 'Helsinki-NLP/opus-mt-en-mt'
model = MarianMTModel.from_pretrained(model_dir)
tokenizer = MarianTokenizer.from_pretrained(model_dir)

# reading the Tatoeba test set
tatoeba_test_path = '/netscratch/dgurgurov/thesis/mt_lrls/data/test_tatoeba.txt'
tatoeba_test_data = pd.read_csv(tatoeba_test_path, sep='\t', header=None, names=['eng', 'mlt', 'eng_sent', 'mlt_sent'])

# tokenizing and predicting
def translate_sentences(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs) # type: ignore
    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return translations

# getting predictions
predictions = translate_sentences(tatoeba_test_data['eng_sent'].tolist())

bleu_metric = load_metric('bleu')

decoded_preds = [pred.split() for pred in predictions]
decoded_labels = [[label.split()] for label in tatoeba_test_data['mlt_sent'].tolist()]

bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels) # type: ignore
print(f"BLEU score: {bleu_score}")

# saving the results
output_df = pd.DataFrame({
    'eng_sent': tatoeba_test_data['eng_sent'],
    'mlt_sent': tatoeba_test_data['mlt_sent'],
    'predicted_mlt_sent': predictions
})
output_df.to_csv('/netscratch/dgurgurov/thesis/mt_lrls/tatoeba_test_results.csv', index=False)

print("Translation and evaluation complete. Results saved to tatoeba_test_results.csv.")
