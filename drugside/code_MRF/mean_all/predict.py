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
# fuse (maxpool + concat)
# =======================
def fuse_features(row):
    vectors = []
    for name, mp in feats.items():
        fp1 = mp.get(row.drug_1_smiles)
        fp2 = mp.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            # if cannot find vector, return None and skip sample
            return None
        pooled = (fp1 + fp2) / 2
        vectors.append(pooled)
    
    return np.concatenate(vectors, axis=0)

# =======================
# build matrix
# =======================
X = []
Y = []
rows = []

for _, row in test_df.iterrows():
    fused = fuse_features(row)
    if fused is None:
        continue
    X.append(fused)
    Y.append(row.mean_reporting_frequency)
    rows.append(row.row_id if "row_id" in row else None)

X = np.array(X)
Y = np.array(Y)

print("Feature size =", X.shape)
print("Test samples =", len(X))

# =======================
# load model + predict
# =======================
model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")

pred = model.predict(X)

# =======================
# output
# =======================
out = pd.DataFrame({
    "row_id": rows,
    "mean_reporting_frequency_true": Y,
    "mean_reporting_frequency_pred": pred
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ Saved pred_fold{fold_id}.csv")

