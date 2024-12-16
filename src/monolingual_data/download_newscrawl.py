import pandas as pd
import nltk
import requests
import gzip
import os
from datetime import datetime
import numpy as np

def download_and_load_data(url, download=False):
    """Download and load the news crawl dataset."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting data processing...")
    filename = "news.2023.en.shuffled.deduped.gz"
    
    if download:
        print("Downloading data...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Download complete")
    
    print("Reading gzipped file...")
    with gzip.open(filename, "rt", encoding='utf-8') as f:
        news_data = f.readlines()
    print(f"Loaded {len(news_data):,} lines of text")
    return news_data

def prepare_and_clean_data(news_data):
    """Convert data to DataFrame and clean it."""
    print("\nPreparing and cleaning data...")
    
    # Convert to DataFrame
    df = pd.DataFrame(news_data, columns=['text'])
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() > 0]
    
    # Calculate sentence lengths
    print("Calculating sentence lengths...")
    df['sentence_length'] = df['text'].apply(lambda x: len(nltk.word_tokenize(str(x))))
    
    # Remove outliers using IQR method
    print("Removing outliers...")
    Q1 = df['sentence_length'].quantile(0.25)
    Q3 = df['sentence_length'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['sentence_length'] >= Q1 - 1.5 * IQR) & 
            (df['sentence_length'] <= Q3 + 1.5 * IQR)]
    
    print(f"Final cleaned dataset size: {len(df):,} sentences")
    return df

def split_and_save_data(df, output_dir):
    """Split the data into different sizes and save to files."""
    print("\nSplitting and saving data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define sizes
    sizes = {
        '25k': 25000,
        '50k': 50000,
        '100k': 100000,
        '250k': 250000,
        '500k': 500000,
        '1m': 1000000,
        '2m': 2000000,
    }
    
    # Save splits
    for name, size in sizes.items():
        if len(df) >= size:
            output_file = f"{output_dir}/news_data_{name}.txt"
            df['text'].head(size).to_csv(output_file, index=False, header=False)
            print(f"Saved {name} split ({size:,} sentences) to {output_file}")
        else:
            print(f"Warning: Not enough data for {name} split (needed {size:,}, have {len(df):,})")

def main():
    # Download NLTK data
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Set up output directory
    output_dir = f"news_splits_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Process data
    url = "https://data.statmt.org/news-crawl/en/news.2023.en.shuffled.deduped.gz"
    news_data = download_and_load_data(url, download=False)  # Set download=True if you need to download the file
    
    # Clean data
    df = prepare_and_clean_data(news_data)
    
    # Split and save
    split_and_save_data(df, output_dir)
    
    print(f"\nProcessing complete. Files saved in: {output_dir}")

if __name__ == "__main__":
    main()