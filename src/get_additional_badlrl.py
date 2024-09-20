import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Translate saved dataset')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# Setting language code from command line argument
language = args.language
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-trained model and tokenizer
model_name = f'Helsinki-NLP/opus-mt-en-{language}'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# For multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

# Custom Dataset class
class TranslationDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Function to collate batches with input validation
def collate_fn(batch):
    # Filter out any non-string entries and empty strings
    valid_batch = [text for text in batch if isinstance(text, str) and text.strip() != '']
    
    if not valid_batch:
        return None  # Return None if there's nothing valid in the batch

    inputs = tokenizer(valid_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs.input_ids, inputs.attention_mask

# Load the filtered dataset from CSV
df = pd.read_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/tinystories_badlrl_filtered.csv')
filtered_subset = df['text'].tolist()

# Create DataLoader for the filtered subset
translation_dataset = TranslationDataset(filtered_subset)
dataloader = DataLoader(translation_dataset, batch_size=64, collate_fn=collate_fn, num_workers=4)

# Translate the English sentences to the low-resource language
translated_sentences = []
for batch in tqdm(dataloader):
    if batch is None:  # Skip any batches that are invalid (if all entries were invalid)
        continue
    
    input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model.module.generate(input_ids, attention_mask=attention_mask) if isinstance(model, torch.nn.DataParallel) else model.generate(input_ids, attention_mask=attention_mask)
    
    batch_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translated_sentences.extend(batch_translated)

# Add the translated sentences to the dataset (as a DataFrame)
df_translations = pd.DataFrame({
    'original_text': filtered_subset,
    'translated_text': translated_sentences,
})

# Save to CSV
df_translations.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_badlrl.csv', index=False)

print("Translation completed and saved to CSV.")