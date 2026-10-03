import pandas as pd
import numpy as np

# File paths
file1 = 'predictions_xgboost.csv'
file2 = 'predictions_lightgbm.csv'
file3 = 'predictions_ori.csv'

# Load and index by 'stimulus'
df1 = pd.read_csv(file1).set_index('stimulus')
df2 = pd.read_csv(file2).set_index('stimulus')
df3 = pd.read_csv(file3).set_index('stimulus')

# Get common stimuli and columns
common_index = df1.index.intersection(df2.index).intersection(df3.index)
common_columns = df1.columns.intersection(df2.columns).intersection(df3.columns)

# Filter and sort
df1_common = df1.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df2_common = df2.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df3_common = df3.loc[common_index, common_columns].sort_index().sort_index(axis=1)

# === Average with equal weight ===
df_weighted = (df1_common + df2_common + df3_common) / 3

# === Quantile normalization ===
df_weighted_sorted = df_weighted.copy()

for col in df_weighted.columns:
    weighted_vals = df_weighted[col].values
    reference_vals = np.sort(df2_common[col].values)  # using df2 as reference

    # Sort and replace
    sorted_weighted_idx = np.argsort(weighted_vals)
    weighted_vals[sorted_weighted_idx] = reference_vals
    df_weighted_sorted[col] = weighted_vals

# Restore 'stimulus' column
df_weighted_sorted = df_weighted_sorted.reset_index()

# Save to CSV
df_weighted_sorted.to_csv('predictions_ori.csv', index=False)
print("✅ Averaged and quantile-normalized predictions saved to predictions.csv")

