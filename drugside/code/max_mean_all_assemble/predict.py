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
# Load test
# =======================
test_df = pd.read_csv(f"test_fold{fold_id}.csv")


# =======================
# Helper: load FP dataset
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
    "rdkit":   load_fp("../../data/features_rdkitfp.csv"),
    "desc":    load_fp("../../data/features_descriptors.csv"),
    "mordred": load_fp("../../data/features_mordred.csv"),
}

print("✔ feature maps ready")


# =======================
# Fuse MAX
# =======================
def fuse_max(row):
    vectors = []
    for fp_map in feats.values():
        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            return None
        vectors.append(np.maximum(fp1, fp2))
    return np.concatenate(vectors, axis=0)


# =======================
# Fuse MEAN
# =======================
def fuse_mean(row):
    vectors = []
    for fp_map in feats.values():
        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            return None
        vectors.append((fp1 + fp2) / 2.0)
    return np.concatenate(vectors, axis=0)


# =======================
# Build test matrices
# =======================
X_max, X_mean, Y, row_ids = [], [], [], []

for _, row in test_df.iterrows():
    fx_max  = fuse_max(row)
    fx_mean = fuse_mean(row)

    if fx_max is None or fx_mean is None:
        continue

    X_max.append(fx_max)
    X_mean.append(fx_mean)

    Y.append(row.PRR)
    row_ids.append(row.row_id)

X_max  = np.array(X_max)
X_mean = np.array(X_mean)
Y      = np.array(Y)


print("Shapes:")
print("X_max  =", X_max.shape)
print("X_mean =", X_mean.shape)


# =======================
# Load models
# =======================
model_max  = CatBoostRegressor()
model_mean = CatBoostRegressor()

model_max.load_model(f"model_max_fold{fold_id}.cbm")
model_mean.load_model(f"model_mean_fold{fold_id}.cbm")

print(f"✔ Loaded max and mean models for fold {fold_id}")


# =======================
# Predict
# =======================
pred_max  = model_max.predict(X_max)
pred_mean = model_mean.predict(X_mean)

# average ensemble (recommended)
pred_avg  = (pred_max + pred_mean) / 2.0


# =======================
# Save output
# =======================
out = pd.DataFrame({
    "row_id": row_ids,
    "PRR_true": Y,
    "PRR_pred_max": pred_max,
    "PRR_pred_mean": pred_mean,
    "PRR_pred": pred_avg,
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ Saved pred_fold{fold_id}.csv")

