import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import requests
import gzip
import os
import time
from datetime import datetime
import numpy as np

def download_and_load_data(url, download=False):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting data download...")
    filename = "news.2023.en.shuffled.deduped.gz"
    
    if download:
        response = requests.get(url)
        
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data downloaded successfully")
        
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading gzipped file...")
    with gzip.open(filename, "rt", encoding='utf-8') as f:
        news_data = f.readlines()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(news_data)} lines of text")
    return news_data

def prepare_data(news_data):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Converting data to DataFrame...")
    df = pd.DataFrame(news_data, columns=['text'])
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cleaning data...")
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() > 0]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating sentence lengths...")
    df['sentence_length'] = df['text'].apply(lambda x: len(nltk.word_tokenize(str(x))))
    
    # Remove outliers
    Q1 = df['sentence_length'].quantile(0.25)
    Q3 = df['sentence_length'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['sentence_length'] >= Q1 - 1.5 * IQR) & 
            (df['sentence_length'] <= Q3 + 1.5 * IQR)]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Final dataset size: {len(df)} sentences")
    return df

def save_detailed_statistics(df, output_dir):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculating and saving detailed statistics...")
    
    # Create statistics directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    stats = df['sentence_length'].describe()
    percentiles = pd.Series({
        f'percentile_{i}': np.percentile(df['sentence_length'], i)
        for i in range(0, 101, 5)
    })
    
    # Additional statistics
    additional_stats = pd.Series({
        'skewness': df['sentence_length'].skew(),
        'kurtosis': df['sentence_length'].kurtosis(),
        'mode': df['sentence_length'].mode().iloc[0],
        'iqr': stats['75%'] - stats['25%'],
        'coefficient_of_variation': df['sentence_length'].std() / df['sentence_length'].mean() * 100
    })
    
    # Combine all statistics
    all_stats = pd.concat([stats, percentiles, additional_stats])
    all_stats.to_csv(f'{output_dir}/detailed_statistics.csv')
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Statistics saved to {output_dir}/detailed_statistics.csv")
    return all_stats

def create_enhanced_plots(df, output_dir):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Creating enhanced visualizations...")
    
    # Set style
    plt.style.use('fivethirtyeight')
    
    # 1. Enhanced Histogram
    plt.figure(figsize=(15, 8))
    sns.histplot(data=df, x='sentence_length', bins=100, kde=True, color='blue', alpha=0.6)
    
    mean_val = df['sentence_length'].mean()
    median_val = df['sentence_length'].median()
    
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
    
    plt.title('Sentence Length Distribution in News Crawl Data\n(with Kernel Density Estimation)', 
              fontsize=14, pad=20)
    plt.xlabel('Number of Words per Sentence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add statistical annotations
    stats_text = f'Standard Deviation: {df["sentence_length"].std():.1f}\n'
    stats_text += f'Skewness: {df["sentence_length"].skew():.2f}\n'
    stats_text += f'Sample Size: {len(df):,}'
    
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                ha='right', va='top', fontsize=10)
    
    plt.savefig(f'{output_dir}/enhanced_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_ttr(df, output_dir):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculating Type-Token Ratio...")
    
    ttr_values = []
    sample_sizes = range(1000, min(len(df), 100000), 10000)
    
    for size in sample_sizes:
        print(f"Processing sample size: {size:,}", end='\r')
        sample_tokens = nltk.word_tokenize(' '.join(df['text'][:size]))
        total_tokens = len(sample_tokens)
        unique_tokens = len(set(sample_tokens))
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        ttr_values.append(ttr)
    
    # Enhanced TTR Plot
    plt.figure(figsize=(12, 6))
    plt.plot(sample_sizes, ttr_values, marker='o', linestyle='-', color='purple', linewidth=2)
    
    plt.title('Type-Token Ratio (TTR) Over Sample Sizes\nMeasuring Lexical Diversity', 
              fontsize=14, pad=20)
    plt.xlabel('Sample Size (Number of Sentences)', fontsize=12)
    plt.ylabel('Type-Token Ratio (TTR)', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.fill_between(sample_sizes, ttr_values, alpha=0.2, color='purple')
    
    # Add trend annotation
    z = np.polyfit(sample_sizes, ttr_values, 1)
    trend = f"Trend: {'↑' if z[0] > 0 else '↓'} {abs(z[0]*1e4):.2f}×10⁻⁴ per sample"
    plt.annotate(trend, xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                ha='left', va='top', fontsize=10)
    
    plt.savefig(f'{output_dir}/enhanced_ttr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save TTR data
    ttr_results = pd.DataFrame({
        'Sample Size': sample_sizes,
        'TTR': ttr_values,
        'Unique_Tokens': [v * s for v, s in zip(ttr_values, sample_sizes)]
    })
    ttr_results.to_csv(f'{output_dir}/ttr_analysis.csv', index=False)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] TTR analysis completed and saved")

def main():
    start_time = time.time()
    
    # Create output directory
    output_dir = f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download NLTK data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Process data
    url = "https://data.statmt.org/news-crawl/en/news.2023.en.shuffled.deduped.gz"
    news_data = download_and_load_data(url)
    df = prepare_data(news_data)
    
    # Generate outputs
    stats = save_detailed_statistics(df, output_dir)
    create_enhanced_plots(df, output_dir)
    calculate_ttr(df, output_dir)
    
    execution_time = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analysis completed in {execution_time:.2f} seconds")
    print(f"Results saved in directory: {output_dir}")

if __name__ == "__main__":
    main()