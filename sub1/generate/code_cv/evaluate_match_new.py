import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. Load predicted fingerprints (74 × 2048)
#    NOTE: Paths are modified to assume files are in the current directory.
# ============================================================
pred_fp = pd.read_csv("predicted_full_fingerprints.csv")
pred_matrix = pred_fp.values.astype(float)
print("Predicted matrix:", pred_matrix.shape)

# ============================================================
# 2. Load test.csv → get stimuli → map to CIDs
# ============================================================
# NOTE: Path modified for 'TASK1_Stimulus_definition.csv'
test = pd.read_csv("../../data/raw/TASK1_Stimulus_definition.csv")
stim_to_cid = dict(zip(test["stimulus"], test["molecule"].astype(str)))

test_csv = pd.read_csv("test.csv")
test_stimuli = test_csv["stimulus"].tolist()

test_cids = [stim_to_cid[s] for s in test_stimuli]
print("Test CIDs:", len(test_cids))

# ============================================================
# 3. Load true fingerprints (CID → 2048 vector)
# ============================================================
# NOTE: Path modified for 'cid_molsig_fp_2048.csv'
real_fp = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")
real_fp["molecule"] = real_fp["molecule"].astype(str)

# Build lookup dictionary: CID → fp (numpy vector)
real_dict = {
    cid: row
    for cid, row in zip(
        real_fp["molecule"],
        real_fp.iloc[:, 2:].values  # already numpy array
    )
}

# ============================================================
# 4. Build real matrix in same order as predicted (74 × 2048)
#    Also keep track of duplicates (same CID appearing twice)
# ============================================================
real_matrix = []
valid_idx = []
invalid_count = 0

for i, cid in enumerate(test_cids):
    if cid in real_dict:
        real_matrix.append(real_dict[cid])
        valid_idx.append(i)
    else:
        # NOTE: Using a matrix of zeros for missing CIDs might affect the
        # similarity calculation for those rows/columns.
        real_matrix.append(np.zeros(2048))
        invalid_count += 1

real_matrix = np.vstack(real_matrix)
print("Real matrix:", real_matrix.shape)
print("Missing CIDs:", invalid_count)

# ============================================================
# 5. Compute similarity matrix (74 × 74)
# ============================================================
def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    # Handle the case where a and/or b are zero vectors (from missing CIDs)
    return (a @ b) / denom if denom != 0 else 0.0

sim_matrix = np.zeros((74, 74))

for i in range(74):
    for j in range(74):
        sim_matrix[i, j] = cosine(pred_matrix[i], real_matrix[j])

# ============================================================
# 6. Compute Top-K accuracy
# ============================================================
top1 = 0
top3 = 0
top5 = 0
top10 = 0

for i in range(74):
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

top1_acc = top1 / 74
top3_acc = top3 / 74
top5_acc = top5 / 74
top10_acc = top10 / 74

# ============================================================
# 7. Compute AUROC: per molecule, then the median
# ============================================================
auc_per_molecule = []

for i in range(74):
    cid_true = test_cids[i]

    # true labels for molecule i: 1 if target j has the same CID as i, 0 otherwise
    # The true labels are defined by the real CID for the predicted molecule at row i.
    true_positions = [j for j, cid in enumerate(test_cids) if cid == cid_true]
    y_true_i = [1 if j in true_positions else 0 for j in range(74)]

    # scores for molecule i: similarity scores from row i of the similarity matrix
    y_score_i = sim_matrix[i].tolist()

    # AUROC is only defined if there is at least one positive (1) and one negative (0) sample.
    # If the current molecule's CID is unique in the test set, it will have a mix of 1s and 0s.
    # If all molecules in the test set had the same CID, len(set(y_true_i)) would be 1.
    if len(set(y_true_i)) > 1:
        auc_i = roc_auc_score(y_true_i, y_score_i)
        print(auc_i)
        auc_per_molecule.append(auc_i)

median_auc = np.median(auc_per_molecule)

# --- Compute the original overall AUC for comparison (optional) ---
y_true_all_orig = []
y_score_all_orig = []
for i in range(74):
    cid_true = test_cids[i]
    true_positions = [j for j, cid in enumerate(test_cids) if cid == cid_true]
    for j in range(74):
        y_true_all_orig.append(1 if j in true_positions else 0)
        y_score_all_orig.append(sim_matrix[i, j])

auc_overall = roc_auc_score(y_true_all_orig, y_score_all_orig)
# -------------------------------------------------------------------


# ============================================================
# 8. Print results (Updated)
# ============================================================
print("\n========== RESULTS (Mean Per-Molecule AUC) ==========")
print(f"Top-1 Accuracy:   {top1_acc:.3f}")
print(f"Top-3 Accuracy:   {top3_acc:.3f}")
print(f"Top-5 Accuracy:   {top5_acc:.3f}")
print(f"Top-10 Accuracy:  {top10_acc:.3f}")
print(f"AUC (Overall):    {auc_overall:.3f}")
print(f"AUC (Median Per-Row): {median_auc:.3f}")
print(f"Number of AUC scores averaged: {len(auc_per_molecule)}")
print("===================================================\n")
