import pandas as pd
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Splitting Tinystories')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
parser.add_argument('--lrl', type=str, required=True, help='Type of LRL to split: goodlrl / badlrl')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
lrl = args.lrl
lang_map = {"sw": "swahili", "mt": "maltese"}

# loading the CSV file
df = pd.read_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_{lrl}_30.csv')

# determining the midpoint
midpoint = len(df) // 2

# splitting the DataFrame into two halves
first_half = df.iloc[:midpoint]
second_half = df.iloc[midpoint:]

# saving the two halves into separate CSV files
first_half.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_{lrl}_1.csv', index=False)
second_half.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/tinystories_{lrl}_2.csv', index=False)

print("Done!")
