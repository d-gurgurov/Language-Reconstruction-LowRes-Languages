import pandas as pd
from bs4 import BeautifulSoup
import argparse

# parsing command line arguments
parser = argparse.ArgumentParser(description='Training a baseline on full data')
parser.add_argument('--language', type=str, required=True, help='Target language code (e.g., "sw" for Swahili)')
args = parser.parse_args()

# setting language code from command line argument
language = args.language
lang_map = {"sw": "swahili", "mt": "maltese", "ga": "irish", "is": "icelandic",
            "tl": "tagalog", "hr": "croatian", "nn": "norwegian"}

# function to parse TMX file and extract parallel sentences
def parse_tmx(file_path, src_lang='en', tgt_lang=language):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
    src_sentences = []
    tgt_sentences = []
    for tu in soup.find_all('tu'):
        src_text = tu.find('tuv', {'xml:lang': src_lang}).seg.get_text()
        tgt_text = tu.find('tuv', {'xml:lang': tgt_lang}).seg.get_text()
        src_sentences.append(src_text)
        tgt_sentences.append(tgt_text)
    return pd.DataFrame({'en': src_sentences, language: tgt_sentences})

# data from paracrawl
tmx_file_path = f'/netscratch/dgurgurov/projects2024/mt_lrls/data/en-{language}.tmx' 
parallel_data = parse_tmx(tmx_file_path)

# splitting the parallel data into two halves
first_half = parallel_data[:len(parallel_data)//2]
second_half = parallel_data[len(parallel_data)//2:]

# saving the two halves to CSV files
first_half.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/first_half.csv', index=False)
second_half.to_csv(f'/netscratch/dgurgurov/projects2024/mt_lrls/data/train_{lang_map[language]}/second_half.csv', index=False)

print("Length of the first half:", len(first_half))
print("Length of the second half:", len(second_half))
