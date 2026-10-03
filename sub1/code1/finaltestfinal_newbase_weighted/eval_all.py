import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import glob
import itertools

# === Gather all prediction files ===
prediction_files = glob.glob("*predictions*.csv")

# === Store pairwise mean correlations ===
results = []

# === Compare each pair ===
for f1, f2 in itertools.combinations(prediction_files, 2):
    df1 = pd.read_csv(f1).set_index("stimulus").sort_index()
    df2 = pd.read_csv(f2).set_index("stimulus").sort_index()

    # Intersect stimuli
    common_stimuli = df1.index.intersection(df2.index)
    df1 = df1.loc[common_stimuli]
    df2 = df2.loc[common_stimuli]

    # Intersect columns
    common_cols = df1.columns.intersection(df2.columns)
    if len(common_cols) == 0 or len(common_stimuli) == 0:
        print(f"⚠️ No overlap between {f1} and {f2}")
        continue

    pearsons = []
    for col in common_cols:
        y1 = df1[col].values
        y2 = df2[col].values

        if len(y1) != len(y2):
            continue

        corr, _ = pearsonr(y1, y2)
        pearsons.append(corr)

    if pearsons:
        mean_corr = np.mean(pearsons)
        results.append((f1, f2, mean_corr))

# === Save results ===
df_result = pd.DataFrame(results, columns=["file1", "file2", "mean_pearson"])
df_result.to_csv("pairwise_mean_correlation.tsv", sep="\t", index=False)

print("✅ Pairwise mean correlations saved to pairwise_mean_correlation.tsv")

