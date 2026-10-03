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
# Fuse (same logic as train)
# =======================
def fuse_features(row):
    vectors = []
    for fp_map in feats.values():
        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)

        if fp1 is None or fp2 is None:
            return None

        max_vec  = np.maximum(fp1, fp2)
        mean_vec = (fp1 + fp2) / 2.0

        vectors.append(max_vec)
        vectors.append(mean_vec)

    return np.concatenate(vectors, axis=0)


# =======================
# Build test matrix
# =======================
X, Y, row_ids = [], [], []

for _, row in test_df.iterrows():
    fused = fuse_features(row)
    if fused is None:
        continue
    X.append(fused)
    Y.append(row.mean_reporting_frequency)
    row_ids.append(row.row_id)

X = np.array(X)
Y = np.array(Y)


# =======================
# Load model
# =======================
model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")


# =======================
# Predict
# =======================
pred = model.predict(X)


# =======================
# Save output
# =======================
out = pd.DataFrame({
    "row_id": row_ids,
    "mean_reporting_frequency_true": Y,
    "mean_reporting_frequency_pred": pred,
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ Saved pred_fold{fold_id}.csv")

