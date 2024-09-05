import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch #type: ignore
from tqdm import tqdm
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Initial BadLRL translation')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# setting language code from command line argument
language = args.language

# arguments
lang_map = {"sw": "swahili", "mt": "maltese"}

# gpu set-up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pre-trained model and tokenizer (Maltese for now)
model_name = f'Helsinki-NLP/opus-mt-en-{language}'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device) #type: ignore

# snippet for multiple gpus
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# translating sentences in batches
def translate_sentences_batch(sentences, tokenizer, model, batch_size=64):
    translated = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.module.generate(**inputs) if isinstance(model, torch.nn.DataParallel) else model.generate(**inputs)
        
        batch_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(batch_translated)
    return translated

# loading the data
first_half = pd.read_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/first_half.csv')
print("Length of the first half:", len(first_half))

# translating the English sentences in the first half to the low-resource language
first_half['bad_lrl'] = translate_sentences_batch(first_half['en'].tolist(), tokenizer, model)

first_half.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/badlrl_first_half.csv', index=False)
