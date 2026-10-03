#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Training fold {fold_id}")

# =======================
# Load train / test data
# =======================
train_df = pd.read_csv(f"train_fold{fold_id}.csv")
test_df  = pd.read_csv(f"test_fold{fold_id}.csv")

# =======================
# Clean PRR & SMILES
# =======================
train_df["PRR"] = pd.to_numeric(train_df["PRR"], errors="coerce")
test_df ["PRR"] = pd.to_numeric(test_df ["PRR"], errors="coerce")

train_df = train_df.dropna(subset=["PRR","drug_1_smiles","drug_2_smiles"])
test_df  = test_df.dropna(subset=["PRR","drug_1_smiles","drug_2_smiles"])

train_df = train_df[(train_df["drug_1_smiles"]!="") & (train_df["drug_2_smiles"]!="")]
test_df  = test_df [(test_df ["drug_1_smiles"]!="") & (test_df ["drug_2_smiles"]!="")]

# =======================
# Load Mordred descriptors
# =======================
mordred_df = pd.read_csv("../../data/features_mordred.csv", low_memory=False)

# Extract SMILES
smiles = mordred_df["SMILES"]

# Numeric descriptors only
numeric = mordred_df.drop(columns=["molecule_id","SMILES"], errors="ignore")
numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

# Mapping: SMILES → Mordred vector
mordred_map = dict(zip(smiles, numeric.values))

# =======================
# Max pool (Mordred only)
# =======================
def meanpool(row):
    fp1 = mordred_map.get(row.drug_1_smiles)
    fp2 = mordred_map.get(row.drug_2_smiles)
    if fp1 is None or fp2 is None:
        return None
    return np.mean((fp1, fp2),axis=0)

# =======================
# Convert to matrix
# =======================
def df_to_matrix(df):
    X = []
    y = []
    for _, row in df.iterrows():
        pooled = meanpool(row)
        if pooled is None:
            continue
        X.append(pooled)
        y.append(row.PRR)
    return np.array(X), np.array(y)

X_train, y_train = df_to_matrix(train_df)
X_test,  y_test  = df_to_matrix(test_df)

print("Train size:", X_train.shape)
print("Test size: ", X_test.shape)

# =======================
# CatBoost model
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

