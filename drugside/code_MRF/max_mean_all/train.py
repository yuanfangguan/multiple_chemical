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
# Clean mean_reporting_frequency & SMILES
# =======================
for df in (train_df, test_df):
    df["mean_reporting_frequency"] = pd.to_numeric(df["mean_reporting_frequency"], errors="coerce")
    df.dropna(subset=["mean_reporting_frequency","drug_1_smiles","drug_2_smiles"], inplace=True)
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
# Fuse features (max + mean pooling)
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
# DF → feature matrix
# =======================
def df_to_matrix(df):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        fused = fuse_features(row)
        if fused is None:
            continue
        X_list.append(fused)
        y_list.append(row.mean_reporting_frequency)
    return np.array(X_list), np.array(y_list)


X_train, y_train = df_to_matrix(train_df)
X_test,  y_test  = df_to_matrix(test_df)


print("Final feature dim =", X_train.shape[1])
print("Train size =", X_train.shape)
print("Test size  =", X_test.shape)


# =======================
# Train CatBoost
# =======================
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=fold_id,
    verbose=200,
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))


# =======================
# Save model
# =======================
model.save_model(f"model_fold{fold_id}.cbm")
print(f"✔ Saved model_fold{fold_id}.cbm")

