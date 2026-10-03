import pandas as pd
import numpy as np

# ==== File paths ====
file1_path = "predictions.csv"  # File to normalize
file2_path = "predictions_sss.csv"  # Reference file

# ==== Load files ====
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Assume first column is ID/stimulus and skip it
id_col = df1.columns[0]
cols = df1.columns[1:]

# ==== Quantile normalize df1 to df2 per column ====
df1_norm = df1.copy()

for col in cols:
    # Get sorted values from reference (df2)
    ref_sorted = np.sort(df2[col].values)

    # Get the ranks of df1's values
    ranks = df1[col].rank(method="min").astype(int) - 1  # 0-based index

    # Map df1's ranks to the reference's sorted values
    df1_norm[col] = ranks.apply(lambda r: ref_sorted[r])

# ==== Save output ====
df1_norm.to_csv("predictions_quantile_normalized.csv", index=False)

print("Quantile normalization complete! Saved to predictions1_quantile_normalized.csv")

