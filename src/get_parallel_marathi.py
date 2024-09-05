import pandas as pd
from datasets import load_dataset

# loading the Samanantar dataset for English to Marathi
dataset = load_dataset('ai4bharat/samanantar', name='mr')

# converting to a pandas DataFrame
df = pd.DataFrame(dataset['train']) #type: ignore

# renaming columns from 'src' to 'en' and 'tgt' to 'mr'
df.rename(columns={'src': 'en', 'tgt': 'mr'}, inplace=True)

# Split the data into two halves
half_idx = len(df) // 2
first_half = df[:half_idx]
second_half = df[half_idx:]

# Save to CSV files
first_half.to_csv('/netscratch/dgurgurov/projects2024/mt_lrls/data/train_marathi/first_half.csv', index=False)
second_half.to_csv('/netscratch/dgurgurov/projects2024/mt_lrls/data/train_marathi/second_half.csv', index=False)
