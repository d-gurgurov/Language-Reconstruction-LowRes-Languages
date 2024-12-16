import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats

def extract_corpus_name(path):
    """Extract corpus name from directory path."""
    dir_name = os.path.basename(os.path.dirname(path))
    return dir_name.split('_analysis')[0]

def read_statistics_files(file_prefixes):
    """Read multiple detailed_statistics and ttr_analysis files."""
    stats_dfs = []
    ttr_dfs = []
    
    for prefix in file_prefixes:
        # Read detailed statistics
        stats_df = pd.read_csv(f"{prefix}detailed_statistics.csv")
        stats_df['corpus'] = extract_corpus_name(prefix)
        stats_dfs.append(stats_df)
        
        # Read TTR analysis
        ttr_df = pd.read_csv(f"{prefix}ttr_analysis.csv")
        ttr_df['corpus'] = extract_corpus_name(prefix)
        ttr_dfs.append(ttr_df)
    
    return pd.concat(stats_dfs), pd.concat(ttr_dfs)

def plot_distributions(stats_df, save_path=None):
    """Create distribution plots with skewness and kurtosis."""
    plt.figure(figsize=(14, 7))
    
    # Color palette for consistency
    colors = sns.color_palette("husl", n_colors=len(stats_df['corpus'].unique()))
    
    for i, corpus in enumerate(stats_df['corpus'].unique()):
        corpus_data = stats_df[stats_df['corpus'] == corpus]
        
        # Extract statistical moments
        mean_row = corpus_data[corpus_data.iloc[:, 0] == 'mean']
        std_row = corpus_data[corpus_data.iloc[:, 0] == 'std']
        skew_row = corpus_data[corpus_data.iloc[:, 0] == 'skewness']
        kurt_row = corpus_data[corpus_data.iloc[:, 0] == 'kurtosis']
        
        if not (mean_row.empty or std_row.empty or skew_row.empty or kurt_row.empty):
            mean = float(mean_row.iloc[0, 1])
            std = float(std_row.iloc[0, 1])
            skewness = float(skew_row.iloc[0, 1])
            kurtosis = float(kurt_row.iloc[0, 1])
            
            # Generate points for the skewed normal distribution
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            
            # Use scipy's skewnorm for the distribution
            # Convert scipy.stats skewness to shape parameter (a)
            # This is an approximation as the relationship is not linear
            a = 4 * np.sign(skewness) * np.log(1 + abs(skewness))
            
            # Generate the skewed distribution
            y = stats.skewnorm.pdf(x, a, loc=mean, scale=std)
            
            # Scale the distribution to match the normal scale
            y = y / np.max(y) * stats.norm.pdf(mean, mean, std)
            
            # Plot the distribution
            plt.plot(x, y, color=colors[i], 
                    label=f'{corpus}\nμ={mean:.1f}, σ={std:.1f}\nskew={skewness:.2f}')
            plt.fill_between(x, 0, y, alpha=0.3, color=colors[i])
            
            # Add vertical line for mean
            plt.axvline(x=mean, ymin=0, ymax=max(y), 
                       color=colors[i], linestyle='--', alpha=0.5)
    
    plt.xlabel('Sentence Length', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Sentence Length Distribution Comparison with Skewness', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_ttr_comparison(ttr_df, save_path=None):
    """Create line plot of TTR analysis."""
    plt.figure(figsize=(15, 8))
    
    for corpus in ttr_df['corpus'].unique():
        corpus_data = ttr_df[ttr_df['corpus'] == corpus]
        plt.plot(corpus_data['Sample Size'], 
                corpus_data['TTR'], 
                label=corpus, 
                marker='o')
    
    plt.xlabel('Sample Size', fontsize=14)
    plt.ylabel('Type-Token Ratio (TTR)', fontsize=14)
    plt.title('TTR Analysis Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# Example usage
file_prefixes = [
    '/netscratch/dgurgurov/projects2024/mt_lrls/corpus_analysis/bookcorpus_analysis_20241206_091349/',
    '/netscratch/dgurgurov/projects2024/mt_lrls/corpus_analysis/news_analysis_20241206_091300/',
    '/netscratch/dgurgurov/projects2024/mt_lrls/corpus_analysis/tinystories_analysis_20241206_093456/'
]

# Read data
stats_df, ttr_df = read_statistics_files(file_prefixes)

# Create and save both visualizations
output_dir = '/netscratch/dgurgurov/projects2024/mt_lrls/corpus_analysis/plots/'
os.makedirs(output_dir, exist_ok=True)

plot_distributions(stats_df, save_path=f'{output_dir}sentence_length_distributions.png')
plot_ttr_comparison(ttr_df, save_path=f'{output_dir}ttr_comparison.png')