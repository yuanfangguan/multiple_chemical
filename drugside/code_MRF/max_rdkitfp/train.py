#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Training fold {fold_id}")

train_df = pd.read_csv(f"train_fold{fold_id}.csv")
test_df  = pd.read_csv(f"test_fold{fold_id}.csv")

# ---- 1. Clean mean_reporting_frequency ----
train_df["mean_reporting_frequency"] = pd.to_numeric(train_df["mean_reporting_frequency"], errors="coerce")
test_df["mean_reporting_frequency"]  = pd.to_numeric(test_df["mean_reporting_frequency"], errors="coerce")

train_df = train_df.dropna(subset=["mean_reporting_frequency", "drug_1_smiles", "drug_2_smiles"])
test_df  = test_df.dropna(subset=["mean_reporting_frequency", "drug_1_smiles", "drug_2_smiles"])

# ---- 2. Remove empty SMILES ----
train_df = train_df[(train_df["drug_1_smiles"] != "") & (train_df["drug_2_smiles"] != "")]
test_df  = test_df[(test_df["drug_1_smiles"] != "") & (test_df["drug_2_smiles"] != "")]

rdkitfp = pd.read_csv("../../data/features_rdkitfp.csv")

rdkitfp_map = dict(zip(
    rdkitfp["SMILES"],
    rdkitfp.drop(columns=["SMILES", "molecule_id"], errors="ignore").values
))

def maxpool(row):
    fp1 = rdkitfp_map.get(row.drug_1_smiles)
    fp2 = rdkitfp_map.get(row.drug_2_smiles)
    if fp1 is None or fp2 is None:
        return None
    return np.maximum(fp1, fp2)

# ---- 3. Convert to matrix, skipping failed fingerprints ----
def df_to_matrix(df):
    fps = []
    y = []
    for _, row in df.iterrows():
        pooled = maxpool(row)
        if pooled is None:
            continue
        fps.append(pooled)
        y.append(row.mean_reporting_frequency)
    return np.array(fps), np.array(y)

X_train, y_train = df_to_matrix(train_df)
X_test, y_test   = df_to_matrix(test_df)

print("Train size:", X_train.shape)
print("Test size:",  X_test.shape)

model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=fold_id,
    verbose=200,
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

model.save_model(f"model_fold{fold_id}.cbm")
print(f"✔ Saved model_fold{fold_id}.cbm")

