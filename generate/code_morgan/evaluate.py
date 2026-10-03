import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. Load predicted fingerprints (N × 166)
# ============================================================
pred_fp = pd.read_csv("predicted_full_fingerprints.csv")
pred_matrix = pred_fp.values.astype(float)
N_samples = pred_matrix.shape[0]

print("Predicted matrix:", pred_matrix.shape)

# ============================================================
# 2. Load test.csv → get stimuli → map to CIDs
# ============================================================
test = pd.read_csv("../../sub1/data/raw/TASK1_Stimulus_definition.csv")
stim_to_cid = dict(zip(test["stimulus"], test["molecule"].astype(str)))

test_csv = pd.read_csv("test.csv")
test_stimuli = test_csv["stimulus"].tolist()

test_cids = [stim_to_cid[s] for s in test_stimuli]
print("Test CIDs:", len(test_cids))

# ============================================================
# 3. Load true MACCS fingerprints (CID → 166 vector)
# ============================================================
# ⭐ FIX: Load MACCS features
real_fp = pd.read_csv("../../sub1/data/processed/features_morgan.csv")
real_fp["molecule"] = real_fp["molecule"].astype(str)

# Select all columns except 'molecule' and 'SMILES'
fp_cols = [col for col in real_fp.columns if col not in ["molecule", "SMILES"]]
MACCS_DIM = len(fp_cols)

# Build lookup dictionary: CID → fp (numpy vector)
real_dict = {
    cid: row
    for cid, row in zip(
        real_fp["molecule"],
        real_fp[fp_cols].values 
    )
}

# ============================================================
# 4. Build real matrix in same order as predicted (N × 166)
# ============================================================
real_matrix = []
invalid_count = 0

for i, cid in enumerate(test_cids):
    if cid in real_dict:
        real_matrix.append(real_dict[cid])
    else:
        # ⭐ FIX: Use 166 for missing CIDs' dimension
        real_matrix.append(np.zeros(MACCS_DIM)) 
        invalid_count += 1

real_matrix = np.vstack(real_matrix)
print("Real matrix:", real_matrix.shape)
print("Missing CIDs:", invalid_count)

# ============================================================
# 5. Compute similarity matrix (N × N)
# ============================================================
def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return (a @ b) / denom if denom != 0 else 0.0

sim_matrix = np.zeros((N_samples, N_samples))

for i in range(N_samples):
    for j in range(N_samples):
        # Now both pred_matrix[i] and real_matrix[j] are (166,)
        sim_matrix[i, j] = cosine(pred_matrix[i], real_matrix[j])

# ============================================================
# 6. Compute Top-K accuracy
# ============================================================
top1, top3, top5, top10 = 0, 0, 0, 0
N = N_samples # Use N_samples as the denominator

for i in range(N):
    rank = np.argsort(sim_matrix[i])[::-1]  # descending

    cid_true = test_cids[i]
    # find all positions where the true CID appears (duplicates!)
    true_positions = [j for j, cid in enumerate(test_cids) if cid == cid_true]

    # check if any match is in top-K
    if any(pos in rank[:1] for pos in true_positions):
        top1 += 1
    if any(pos in rank[:3] for pos in true_positions):
        top3 += 1
    if any(pos in rank[:5] for pos in true_positions):
        top5 += 1
    if any(pos in rank[:10] for pos in true_positions):
        top10 += 1

top1_acc = top1 / N
top3_acc = top3 / N
top5_acc = top5 / N
top10_acc = top10 / N

# ============================================================
# 7. Compute AUROC: per molecule, then the median
# ============================================================
auc_per_molecule = []

for i in range(N):
    cid_true = test_cids[i]

    # true labels
    true_positions = [j for j, cid in enumerate(test_cids) if cid == cid_true]
    y_true_i = [1 if j in true_positions else 0 for j in range(N)]

    # scores
    y_score_i = sim_matrix[i].tolist()

    if len(set(y_true_i)) > 1:
        auc_i = roc_auc_score(y_true_i, y_score_i)
        auc_per_molecule.append(auc_i)

median_auc = np.median(auc_per_molecule)

# --- Compute the original overall AUC for comparison (optional) ---
y_true_all_orig = []
y_score_all_orig = []
for i in range(N):
    cid_true = test_cids[i]
    true_positions = [j for j, cid in enumerate(test_cids) if cid == cid_true]
    for j in range(N):
        y_true_all_orig.append(1 if j in true_positions else 0)
        y_score_all_orig.append(sim_matrix[i, j])

auc_overall = roc_auc_score(y_true_all_orig, y_score_all_orig)
# -------------------------------------------------------------------

# ============================================================
# 8. Save and Print results
# ============================================================
metrics = {
    "Metric": ["Top-1 Accuracy", "Top-3 Accuracy", "Top-5 Accuracy", "Top-10 Accuracy", "AUC (Overall)", "AUC (Median Per-Row)"],
    "Value": [top1_acc, top3_acc, top5_acc, top10_acc, auc_overall, median_auc]
}
summary_df = pd.DataFrame(metrics)
summary_df.to_csv("evaluation_summary.csv", index=False)
np.save("similarity_matrix_morgan.npy", sim_matrix)

print("\n==============================")
print("Evaluation completed!")
print("Saved summary metrics to: evaluation_summary_morgan.csv")
print("==============================\n")

print("========== RESULTS (MACCS Fingerprints) ==========")
print(f"Top-1 Accuracy:   {top1_acc:.3f}")
print(f"Top-3 Accuracy:   {top3_acc:.3f}")
print(f"Top-5 Accuracy:   {top5_acc:.3f}")
print(f"Top-10 Accuracy:  {top10_acc:.3f}")
print(f"AUC (Overall):    {auc_overall:.3f}")
print(f"AUC (Median Per-Row): {median_auc:.3f}")
print(f"Number of AUC scores averaged: {len(auc_per_molecule)}")
print("===================================================\n")
