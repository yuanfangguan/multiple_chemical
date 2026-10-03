#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Predicting fold {fold_id}")

# =======================
# Load test data
# =======================
test_df = pd.read_csv(f"test_fold{fold_id}.csv")

# =======================
# Helper to load feature maps
# =======================
def load_fp(path, drop_cols=["SMILES","molecule_id"]):
    df = pd.read_csv(path, low_memory=False)
    smiles = df["SMILES"]
    numeric = df.drop(columns=drop_cols, errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return dict(zip(smiles, numeric.values))

print("Loading feature maps...")

feats = {}
feats["morgan"]      = load_fp("../../data/features_morgan.csv")
feats["maccs"]       = load_fp("../../data/features_maccs.csv")
feats["rdkitfp"]     = load_fp("../../data/features_rdkitfp.csv")
feats["desc"]        = load_fp("../../data/features_descriptors.csv")
feats["mordred"]     = load_fp("../../data/features_mordred.csv")

print("✔ feature maps ready")

# =======================
# Single-drug feature extraction
# =======================
def get_features_single(smiles):
    vectors = []
    for name, mp in feats.items():
        fp = mp.get(smiles)
        if fp is None:
            return None
        vectors.append(fp)
    return np.concatenate(vectors, axis=0)

# =======================
# Build feature matrices (vectorized)
# =======================
X1 = []
X2 = []
rows = []
true_vals = []

for _, row in test_df.iterrows():

    f1 = get_features_single(row.drug_1_smiles)
    f2 = get_features_single(row.drug_2_smiles)

    if f1 is None or f2 is None:
        continue

    X1.append(f1)
    X2.append(f2)

    rows.append(row.row_id if "row_id" in row else None)
    true_vals.append(row.mean_reporting_frequency)

X1 = np.array(X1)
X2 = np.array(X2)

print("Feature shapes:")
print("X1 =", X1.shape)
print("X2 =", X2.shape)
print("Samples =", len(X1))

# =======================
# Load model + predict (batched)
# =======================
model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")

pred1 = model.predict(X1)
pred2 = model.predict(X2)

pred_avg = 0.5 * (pred1 + pred2)

# =======================
# Output
# =======================
out = pd.DataFrame({
    "row_id": rows,
    "mean_reporting_frequency_true": true_vals,
    "mean_reporting_frequency_pred": pred_avg
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ Saved pred_fold{fold_id}.csv")

