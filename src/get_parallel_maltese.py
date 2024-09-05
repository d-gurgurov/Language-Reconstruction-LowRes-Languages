import pandas as pd
from bs4 import BeautifulSoup

# function to parse TMX file and extract parallel sentences
def parse_tmx(file_path, src_lang='en', tgt_lang='mt'):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
    src_sentences = []
    tgt_sentences = []
    for tu in soup.find_all('tu'):
        src_text = tu.find('tuv', {'xml:lang': src_lang}).seg.get_text()
        tgt_text = tu.find('tuv', {'xml:lang': tgt_lang}).seg.get_text()
        src_sentences.append(src_text)
        tgt_sentences.append(tgt_text)
    return pd.DataFrame({'en': src_sentences, 'mt': tgt_sentences})

# data from paracrawl
tmx_file_path = '/netscratch/dgurgurov/thesis/mt_lrls/data/en-mt.tmx' 
parallel_data = parse_tmx(tmx_file_path)

# splitting the parallel data into two halves
first_half = parallel_data[:len(parallel_data)//2]
second_half = parallel_data[len(parallel_data)//2:]

# saving the two halves to CSV files
first_half.to_csv('/netscratch/dgurgurov/projects2024/mt_lrls/data/train_maltese/first_half.csv', index=False)
second_half.to_csv('/netscratch/dgurgurov/projects2024/mt_lrls/data/train_maltese/second_half.csv', index=False)

print("Length of the first half:", len(first_half))
print("Length of the second half:", len(second_half))
