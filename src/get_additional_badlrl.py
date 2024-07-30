import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pre-trained model and tokenizer (Maltese for now)
model_name = 'Helsinki-NLP/opus-mt-en-mt'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

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

# load the dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories", split='train')
print("Length of the dataset:", len(dataset))

# calculate 20% of the dataset
subset_size = int(len(dataset) * 0.2)

# create a 20% subset
dataset = dataset.select(range(subset_size))

# extract the English sentences
english_sentences = dataset['text']

# create DataLoader
translation_dataset = TranslationDataset(english_sentences)
dataloader = DataLoader(translation_dataset, batch_size=128, collate_fn=collate_fn, num_workers=4)

# translate the English sentences to the low-resource language
translated_sentences = translate_sentences_batch(dataloader, model)

# add the translated sentences to the dataset
dataset = dataset.add_column('translated_text', translated_sentences)

# convert to pandas DataFrame and save to CSV
df = pd.DataFrame(dataset)
df.to_csv('/netscratch/dgurgurov/thesis/mt_lrls/results/tinystories_badlrl.csv', index=False)

print("Translation completed and saved to CSV.")
