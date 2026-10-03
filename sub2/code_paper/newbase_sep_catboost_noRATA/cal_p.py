import pandas as pd
from scipy import stats
import os

# Define paths
folder1_path = "evaluation.tsv.4"
folder2_path = "../new_nonparrallel_noRATA/evaluation.tsv.4"

# Load the data
df1 = pd.read_csv(folder1_path, sep='\t')
df2 = pd.read_csv(folder2_path, sep='\t')

# Ensure they are sorted by label so we compare the same things
df1 = df1.sort_values('label').reset_index(drop=True)
df2 = df2.sort_values('label').reset_index(drop=True)

# Merge to ensure alignment
merged = pd.merge(df1, df2, on='label', suffixes=('_f1', '_f2'))

def calculate_stats(metric_name):
    # Extract the two arrays
    vec1 = merged[f'{metric_name}_f1']
    vec2 = merged[f'{metric_name}_f2']
    
    # 1. Wilcoxon Signed-Rank Test (Non-parametric paired test)
    # Good for comparing "Which folder is generally better?"
    stat, p_val = stats.wilcoxon(vec1, vec2)
    
    # 2. Basic descriptives
    mean1 = vec1.mean()
    mean2 = vec2.mean()
    win_rate = (vec2 > vec1).mean() * 100 # How often folder 2 beats folder 1
    
    print(f"--- Results for {metric_name.upper()} ---")
    print(f"Folder 1 Mean: {mean1:.4f}")
    print(f"Folder 2 Mean: {mean2:.4f}")
    print(f"Difference:    {mean2 - mean1:.4f}")
    print(f"Win Rate (F2): {win_rate:.1f}%")
    print(f"P-value:       {p_val:.6f}")
    if p_val < 0.05:
        print("RESULT: Statistically Significant Difference")
    else:
        print("RESULT: No Significant Difference")
    print("\n")

# Run for both Pearson and Cosine
calculate_stats('pearson')
calculate_stats('cosine')
