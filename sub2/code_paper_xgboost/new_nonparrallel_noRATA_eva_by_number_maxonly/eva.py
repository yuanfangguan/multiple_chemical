import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances
import os

# === File paths ===
TEST_PATH = "test.csv"
PRED_PATH = "predictions.csv"
OUTPUT_DIR = "evaluation_groups"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_pred = pd.read_csv(PRED_PATH)

# Align on stimulus column
df_test = df_test.set_index("stimulus").sort_index()
df_pred = df_pred.set_index("stimulus").sort_index()

# Sanity check: common stimuli
common_stimuli = df_test.index.intersection(df_pred.index)
df_test = df_test.loc[common_stimuli]
df_pred = df_pred.loc[common_stimuli]

# Extract number of components if present
if "num_components" not in df_pred.columns:
    raise ValueError("❌ 'num_components' column not found in predictions.csv — please re-run prediction script with this field included.")

num_components = df_pred["num_components"]
df_pred = df_pred.drop(columns=["num_components"])

# Ensure label columns match
labels = [col for col in df_pred.columns if col in df_test.columns]
if not labels:
    raise ValueError("❌ No matching labels found between test and prediction columns.")

# === Helper function: run evaluation for a subset ===
def evaluate_subset(df_true, df_pred, output_path):
    results = []
    for label in labels:
        y_true = df_true[label].values
        y_pred = df_pred[label].values
        if len(y_true) != len(y_pred):
            continue
        pearson_corr = pearsonr(y_true, y_pred)[0]
        cos_dist = cosine_distances(y_true.reshape(1, -1), y_pred.reshape(1, -1))[0, 0]
        results.append((label, pearson_corr, cos_dist))

    # Global metrics
    all_true = df_true[labels].values.flatten()
    all_pred = df_pred[labels].values.flatten()
    if len(all_true) == len(all_pred):
        global_pearson = pearsonr(all_true, all_pred)[0]
        global_cosine = cosine_distances(all_true.reshape(1, -1), all_pred.reshape(1, -1))[0, 0]
        results.append(("ALL", global_pearson, global_cosine))

    # Mean across labels
    if results:
        pearson_vals = [r[1] for r in results if r[0] not in ["ALL", "MEAN"]]
        cosine_vals = [r[2] for r in results if r[0] not in ["ALL", "MEAN"]]
        if pearson_vals and cosine_vals:
            mean_pearson = np.mean(pearson_vals)
            mean_cosine = np.mean(cosine_vals)
            results.append(("MEAN", mean_pearson, mean_cosine))

    df_result = pd.DataFrame(results, columns=["label", "pearson", "cosine"])
    df_result.to_csv(output_path, sep="\t", index=False)
    print(f"✅ Saved evaluation to {output_path}")

# === Step 1: Full dataset evaluation ===
evaluate_subset(df_test, df_pred, os.path.join(OUTPUT_DIR, "evaluation_all.tsv"))

# === Step 2: Grouped by num_components ===
unique_groups = sorted(df_pred.index.map(num_components).unique())
print(f"\n=== Found {len(unique_groups)} component groups: {unique_groups} ===")

for n in unique_groups:
    subset_idx = num_components[num_components == n].index
    df_true_sub = df_test.loc[subset_idx]
    df_pred_sub = df_pred.loc[subset_idx]
    if len(df_true_sub) < 2:
        print(f"⚠️ Skipping group {n} (only {len(df_true_sub)} samples)")
        continue

    out_path = os.path.join(OUTPUT_DIR, f"evaluation_numcomp_{n}.tsv")
    print(f"→ Evaluating group with {n} components ({len(df_true_sub)} stimuli)...")
    evaluate_subset(df_true_sub, df_pred_sub, out_path)

