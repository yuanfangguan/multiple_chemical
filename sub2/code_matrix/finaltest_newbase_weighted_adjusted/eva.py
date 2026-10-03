import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances

# === File paths ===
TEST_PATH = "test.csv"
PRED_PATH = "predictions_quantile_normalized.csv"
OUTPUT_TSV = "evaluation.tsv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_pred = pd.read_csv(PRED_PATH)

# Align on stimulus column
df_test = df_test.set_index("stimulus").sort_index()
df_pred = df_pred.set_index("stimulus").sort_index()

# Sanity check: stimuli
common_stimuli = df_test.index.intersection(df_pred.index)
df_test = df_test.loc[common_stimuli]
df_pred = df_pred.loc[common_stimuli]

# Ensure columns match
labels = [col for col in df_pred.columns if col in df_test.columns]

results = []

for label in labels:
    y_true = df_test[label].values
    y_pred = df_pred[label].values

    if len(y_true) != len(y_pred):
        print(f"❌ Length mismatch for label '{label}': y_true={len(y_true)}, y_pred={len(y_pred)}")
        continue  # Skip this label to avoid crashing

    pearson_corr = pearsonr(y_true, y_pred)[0]
    cos_dist = cosine_distances(y_true.reshape(1, -1), y_pred.reshape(1, -1))[0, 0]

    results.append((label, pearson_corr, cos_dist))

# Global (ALL) flattened
all_true = df_test[labels].values.flatten()
all_pred = df_pred[labels].values.flatten()

if len(all_true) != len(all_pred):
    print(f"❌ Global mismatch: all_true={len(all_true)}, all_pred={len(all_pred)}")
else:
    global_pearson = pearsonr(all_true, all_pred)[0]
    global_cosine = cosine_distances(all_true.reshape(1, -1), all_pred.reshape(1, -1))[0, 0]
    results.append(("ALL", global_pearson, global_cosine))

# Mean across labels (excluding ALL)
if results:
    pearson_vals = [r[1] for r in results if r[0] != "ALL"]
    cosine_vals = [r[2] for r in results if r[0] != "ALL"]
    if pearson_vals and cosine_vals:
        mean_pearson = np.mean(pearson_vals)
        mean_cosine = np.mean(cosine_vals)
        results.append(("MEAN", mean_pearson, mean_cosine))

# Save to TSV
df_result = pd.DataFrame(results, columns=["label", "pearson", "cosine"])
df_result.to_csv(OUTPUT_TSV, sep="\t", index=False)

print(f"✅ Evaluation results saved to {OUTPUT_TSV}")

