import requests
import zipfile
import os
import pandas as pd
import gzip
import sys

# Step 1: Download the dataset
url = "http://data.statmt.org/gourmet/corpora/GoURMET-crawled.en-sw.zip"
zip_path = "/netscratch/dgurgurov/projects2024/mt_lrls/results/GoURMET-crawled.en-sw.zip"

# downloading the zip file
response = requests.get(url)
with open(zip_path, 'wb') as file:
   file.write(response.content)

# Step 2: Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
   zip_ref.extractall("./results")

# path where files are extracted
extracted_path = "./results"

# Step 3: Load the gzipped English and Swahili files
en_file = os.path.join(extracted_path, "GoURMET-crawled.sw-en.en.gz")
sw_file = os.path.join(extracted_path, "GoURMET-crawled.sw-es.sw.gz")

# reading and decoding the gzipped files
with gzip.open(en_file, "rt", encoding="utf-8") as f:
    en_lines = f.readlines()


with gzip.open(sw_file, "rt", encoding="utf-8") as f:
    sw_lines = f.readlines()


# Step 4: Create a DataFrame
data = {
    "en": [line.strip() for line in en_lines],
    "sw": [line.strip() for line in sw_lines],
}

df = pd.DataFrame(data)

# Step 5: Split the data into two halves
half_idx = len(df) // 2
first_half = df[:half_idx]
second_half = df[half_idx:]

# Step 6: Save to CSV files
first_half.to_csv('/netscratch/dgurgurov/projects2024/mt_lrls/data/train_swahili/first_half.csv', index=False)
second_half.to_csv('/netscratch/dgurgurov/projects2024/mt_lrls/data/train_swahili/second_half.csv', index=False)
