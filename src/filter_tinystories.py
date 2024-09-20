from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Filter and save dataset')
parser.add_argument('--language', type=str, required=False, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# Setting language code from command line argument
language = args.language
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}

# Load the dataset from Hugging Face
dataset = load_dataset("roneneldan/TinyStories", split='train')
print("Length of the dataset:", len(dataset))

# Extract the English sentences
english_sentences = dataset['text']

# Filter out sentences with more than 512 tokens
model_name = f'FacebookAI/xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

filtered_sentences = []
for sentence in tqdm(english_sentences, desc="Filtering sentences", unit="sentence"):
    if len(tokenizer.encode(sentence)) <= 512:
        filtered_sentences.append(sentence)

print("Length after filtering:", len(filtered_sentences))

# Calculate 20% of the filtered dataset
subset_size = int(len(filtered_sentences) * 0.2)

# Create a 20% subset from the filtered sentences
filtered_subset = filtered_sentences[:subset_size]

# Save the filtered subset to a CSV file
df = pd.DataFrame({'text': filtered_subset})
df.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/tinystories_badlrl_filtered.csv', index=False)

print("Filtered dataset saved to CSV.")