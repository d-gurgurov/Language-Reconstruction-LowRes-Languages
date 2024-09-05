import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from tqdm import tqdm
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Translating BadLRL to GoodLRL')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
lang_map = {"sw": "swahili", "mt": "maltese"}

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading the trained model and tokenizer
model_path = f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/reconstruction_30'
tokenizer_path = f'/netscratch/dgurgurov/projects2024/mt_lrls/models/{lang_map[language]}/reconstruction_30'
tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
model = MarianMTModel.from_pretrained(model_path).to(device) # type: ignore

torch.compile(model)

# for multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# custom Dataset class
class TranslationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# function to collate batches
def collate_fn(batch):
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs.input_ids, inputs.attention_mask

# function to translate sentences in batches
def translate_sentences_batch(dataloader, model):
    model.eval()
    translated = []
    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.module.generate(input_ids, attention_mask=attention_mask) if isinstance(model, torch.nn.DataParallel) else model.generate(input_ids, attention_mask=attention_mask)
        
        batch_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(batch_translated)
    return translated

# loading the synthetic dataset
synthetic_data = pd.read_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_badlrl.csv')

# extracting the synthetic sentences
synthetic_sentences = synthetic_data['translated_text'].tolist()

# creating DataLoader
translation_dataset = TranslationDataset(synthetic_sentences)
dataloader = DataLoader(translation_dataset, batch_size=128, collate_fn=collate_fn, num_workers=4)

# translating the synthetic sentences
translated_sentences = translate_sentences_batch(dataloader, model)

# adding the translated sentences to the synthetic dataset
synthetic_data['good_lrl'] = translated_sentences

# Save the translated dataset to CSV
synthetic_data.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_goodlrl_30.csv', index=False)

print(f"Translation completed and saved to '/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_goodlrl_30.csv'")
