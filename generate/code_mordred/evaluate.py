import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# 1. Load predicted Mordred descriptors
# ============================================================
pred_df = pd.read_csv("predicted_mordred_descriptors.csv")
pred_matrix = pred_df.values.astype(float)

N_samples, D_pred = pred_matrix.shape
print("Predicted matrix:", pred_matrix.shape)

# ============================================================
# 2. Load valid descriptor names used during training
# ============================================================
print("Loading valid Mordred descriptor list...")

valid_cols = pd.read_csv("valid_mordred_cols.csv", header=None)[0].tolist()
D_valid = len(valid_cols)

print(f"Valid Mordred descriptor count: {D_valid}")

if D_pred != D_valid:
    print(f"\n❌ ERROR: Prediction dim {D_pred} != valid dim {D_valid}")
    exit()

# ============================================================
# 3. Load test.csv → map stimulus → molecule (CID)
# ============================================================
stim_df = pd.read_csv("../../sub1/data/raw/TASK1_Stimulus_definition.csv")
stim_to_cid = dict(zip(stim_df["stimulus"], stim_df["molecule"].astype(str)))

test_csv = pd.read_csv("test.csv")
test_stimuli = test_csv["stimulus"].tolist()

test_cids = [stim_to_cid[s] for s in test_stimuli]
print("Test CIDs:", len(test_cids))

# ============================================================
# 4. Load TRUE Mordred descriptors and CLEAN them
# ============================================================
print("\nLoading & cleaning TRUE Mordred descriptors...")

true_md = pd.read_csv("../../sub1/data/raw/Mordred_Descriptors.csv",
                      encoding="latin-1")

true_md["molecule"] = true_md["molecule"].astype(str)

# Drop SMILES if present
if "SMILES" in true_md.columns:
    true_md = true_md.drop(columns=["SMILES"])

# Clean descriptors
descriptor_cols = [c for c in true_md.columns if c != "molecule"]

invalid_tokens = [
    "", " ", ".", "?", "None", "nan", "NaN",
    "Infinity", "-Infinity", "inf", "-inf",
    "--", "CalcError", "ERROR", "N/A"
]

true_md[descriptor_cols] = true_md[descriptor_cols].replace(invalid_tokens, np.nan)

# Convert every descriptor to numeric
for c in descriptor_cols:
    true_md[c] = pd.to_numeric(true_md[c], errors="coerce")

# Fill NaN with column means
true_md[descriptor_cols] = true_md[descriptor_cols].fillna(true_md[descriptor_cols].mean())

print("TRUE matrix loaded:", true_md.shape)

# ============================================================
# 5. *** ALIGN TRUE DESCRIPTORS to the SAME columns as used in training ***
# ============================================================
missing_cols = [c for c in valid_cols if c not in descriptor_cols]

if len(missing_cols) > 0:
    print("\n❌ ERROR: Some valid training descriptors are missing in true Mordred file!")
    print("Missing (first 20 shown):", missing_cols[:20])
    exit()

# Subselect the exact columns (in exact order!)
true_md = true_md[["molecule"] + valid_cols]

print("Aligned TRUE descriptor matrix:", true_md.shape)

# ============================================================
# 6. Build REAL matrix aligned to predicted matrix order
# ============================================================
true_dict = {
    cid: row 
    for cid, row in zip(true_md["molecule"], true_md[valid_cols].values)
}

true_matrix = []
missing = 0

for cid in test_cids:
    if cid in true_dict:
        true_matrix.append(true_dict[cid])
    else:
        true_matrix.append(np.zeros(D_valid))  # fallback
        missing += 1

true_matrix = np.vstack(true_matrix)

print("Final TRUE matrix shape:", true_matrix.shape)
print("Missing CIDs:", missing)

# ============================================================
# 7. Compute similarity matrix (N × N)
# ============================================================
print("\nComputing similarity matrix...")

def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return (a @ b) / denom if denom != 0 else 0.0

sim_matrix = np.zeros((N_samples, N_samples))

for i in range(N_samples):
    for j in range(N_samples):
        sim_matrix[i, j] = cosine(pred_matrix[i], true_matrix[j])

# ============================================================
# 8. Compute Top-K accuracy
# ============================================================
print("Computing Top-K accuracy...")

top1 = top3 = top5 = top10 = 0
N = N_samples

for i in range(N):
    rank = np.argsort(sim_matrix[i])[::-1]

    cid_true = test_cids[i]
    true_pos = [j for j, cid in enumerate(test_cids) if cid == cid_true]

    if any(p in rank[:1]  for p in true_pos): top1  += 1
    if any(p in rank[:3]  for p in true_pos): top3  += 1
    if any(p in rank[:5]  for p in true_pos): top5  += 1
    if any(p in rank[:10] for p in true_pos): top10 += 1

top1_acc  = top1  / N
top3_acc  = top3  / N
top5_acc  = top5  / N
top10_acc = top10 / N

# ============================================================
# 9. Compute AUROC (overall + per-row median)
# ============================================================
print("Computing AUROC...")

auc_list = []

for i in range(N):
    cid_true = test_cids[i]
    true_pos = [j for j, cid in enumerate(test_cids) if cid == cid_true]

    y_true  = [1 if j in true_pos else 0 for j in range(N)]
    y_score = sim_matrix[i].tolist()

    if len(set(y_true)) > 1:
        auc_list.append(roc_auc_score(y_true, y_score))

median_auc = np.median(auc_list)

# Flatten overall AUROC
y_true_all = []
y_score_all = []

for i in range(N):
    cid_true = test_cids[i]
    true_pos = [j for j, cid in enumerate(test_cids) if cid == cid_true]
    for j in range(N):
        y_true_all.append(1 if j in true_pos else 0)
        y_score_all.append(sim_matrix[i, j])

auc_overall = roc_auc_score(y_true_all, y_score_all)

# ============================================================
# 10. Save results
# ============================================================
summary = pd.DataFrame({
    "Metric": [
        "Top-1 Accuracy",
        "Top-3 Accuracy",
        "Top-5 Accuracy",
        "Top-10 Accuracy",
        "AUC (Overall)",
        "AUC (Median Per-Row)"
    ],
    "Value": [
        top1_acc,
        top3_acc,
        top5_acc,
        top10_acc,
        auc_overall,
        median_auc
    ]
})

summary.to_csv("evaluation_summary.csv", index=False)
np.save("similarity_matrix_mordred.npy", sim_matrix)

print("\n==============================")
print("MORDRED Evaluation completed!")
print("Saved summary metrics to: evaluation_summary_mordred.csv")
print("==============================\n")

print("========== RESULTS (Mordred Descriptors) ==========")
print(f"Top-1 Accuracy:          {top1_acc:.3f}")
print(f"Top-3 Accuracy:          {top3_acc:.3f}")
print(f"Top-5 Accuracy:          {top5_acc:.3f}")
print(f"Top-10 Accuracy:         {top10_acc:.3f}")
print(f"AUC (Overall):           {auc_overall:.3f}")
print(f"AUC (Median Per-Row):    {median_auc:.3f}")
print(f"AUC count:               {len(auc_list)}")
print("===================================================\n")

