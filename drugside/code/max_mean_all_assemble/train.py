#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor


# =======================
# Args
# =======================
fold_id = int(sys.argv[1])
print(f"Training fold {fold_id}")


# =======================
# Load train/test data
# =======================
train_df = pd.read_csv(f"train_fold{fold_id}.csv")
test_df  = pd.read_csv(f"test_fold{fold_id}.csv")


# =======================
# Clean PRR & SMILES
# =======================
for df in (train_df, test_df):
    df["PRR"] = pd.to_numeric(df["PRR"], errors="coerce")
    df.dropna(subset=["PRR","drug_1_smiles","drug_2_smiles"], inplace=True)
    df = df[(df["drug_1_smiles"]!="") & (df["drug_2_smiles"]!="")]


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
# Fuse features - MAX only
# =======================
def fuse_max(row):
    vectors = []
    for fp_map in feats.values():

        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            return None

        max_vec = np.maximum(fp1, fp2)
        vectors.append(max_vec)

    return np.concatenate(vectors, axis=0)


# =======================
# Fuse features - MEAN only
# =======================
def fuse_mean(row):
    vectors = []
    for fp_map in feats.values():

        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            return None

        mean_vec = (fp1 + fp2) / 2.0
        vectors.append(mean_vec)

    return np.concatenate(vectors, axis=0)


# =======================
# DF → feature matrix
# =======================
def df_to_matrix(df, fuse_fn):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        fused = fuse_fn(row)
        if fused is None:
            continue
        X_list.append(fused)
        y_list.append(row.PRR)
    return np.array(X_list), np.array(y_list)


# ------- Build MAX dataset
X_train_max, y_train = df_to_matrix(train_df, fuse_max)
X_test_max,  y_test  = df_to_matrix(test_df,  fuse_max)

# ------- Build MEAN dataset
X_train_mean, _ = df_to_matrix(train_df, fuse_mean)
X_test_mean,  _ = df_to_matrix(test_df,  fuse_mean)


print("MAX feature dim  =", X_train_max.shape[1])
print("MEAN feature dim =", X_train_mean.shape[1])


# =======================
# Train CatBoost (MAX)
# =======================
model_max = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=fold_id,
    verbose=200,
)

model_max.fit(X_train_max, y_train, eval_set=(X_test_max, y_test))
model_max.save_model(f"model_max_fold{fold_id}.cbm")
print(f"✔ Saved model_max_fold{fold_id}.cbm")


# =======================
# Train CatBoost (MEAN)
# =======================
model_mean = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=fold_id,
    verbose=200,
)

model_mean.fit(X_train_mean, y_train, eval_set=(X_test_mean, y_test))
model_mean.save_model(f"model_mean_fold{fold_id}.cbm")
print(f"✔ Saved model_mean_fold{fold_id}.cbm")

