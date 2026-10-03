#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

# =======================
# Args
# =======================
fold_id = int(sys.argv[1])
print(f"Predicting fold {fold_id}")

# =======================
# Load test data
# =======================
test_df = pd.read_csv(f"test_missing_pairs.csv")

# =======================
# Helper: load feature maps
# =======================
def load_fp(path, drop_cols=["SMILES","molecule_id"]):
    df = pd.read_csv(path, low_memory=False)
    smiles = df["SMILES"]
    numeric = df.drop(columns=drop_cols, errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return dict(zip(smiles, numeric.values))

print("Loading feature maps...")

feats = {
    "morgan":  load_fp("../../data/features_morgan.csv"),
    "maccs":   load_fp("../../data/features_maccs.csv"),
    "rdkitfp": load_fp("../../data/features_rdkitfp.csv"),
    "desc":    load_fp("../../data/features_descriptors.csv"),
    "mordred": load_fp("../../data/features_mordred.csv"),
}

print("✔ feature maps ready")

# =======================
# Single-drug feature extraction
# =======================
def get_features_single(smiles):
    vectors = []
    for _, fp_map in feats.items():
        fp = fp_map.get(smiles)
        if fp is None:
            return None
        vectors.append(fp)
    return np.concatenate(vectors, axis=0)

# =======================
# Build concatenated feature matrix (one example per pair)
# =======================
X = []
row_ids = []
true_vals = []

for _, row in test_df.iterrows():

    f1 = get_features_single(row.drug_1_smiles)
    f2 = get_features_single(row.drug_2_smiles)

    # skip if either is missing
    if f1 is None or f2 is None:
        continue

    # 🔥 NEW: concatenation, same as training
    fused = np.concatenate([f1, f2], axis=0)

    X.append(fused)
    true_vals.append(row.mean_reporting_frequency)

    # row_id safe extraction
    row_ids.append(row.get("row_id", None))

X = np.array(X)

print("Feature shape =", X.shape)
print("Samples =", len(X))


# =======================
# Load model + predict
# =======================
model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")

pred = model.predict(X)


# =======================
# Output
# =======================
out = pd.DataFrame({
    "row_id": row_ids,
    "mean_reporting_frequency_true": true_vals,
    "mean_reporting_frequency_pred": pred,
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ Saved pred_fold{fold_id}.csv")

