#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Predicting fold {fold_id}")

test_df = pd.read_csv(f"test_fold{fold_id}.csv")

# Load Mordred
mordred_df = pd.read_csv("../../data/features_mordred.csv", low_memory=False)
smiles = mordred_df["SMILES"]
numeric = mordred_df.drop(columns=["molecule_id","SMILES"], errors="ignore")
numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
mordred_map = dict(zip(smiles, numeric.values))

def meanpool(row):
    fp1 = mordred_map.get(row.drug_1_smiles)
    fp2 = mordred_map.get(row.drug_2_smiles)
    if fp1 is None or fp2 is None:
        return None
    return np.mean((fp1, fp2),axis=0)

X = []
Y = []

for _, row in test_df.iterrows():
    pooled = meanpool(row)
    if pooled is None:
        continue
    X.append(pooled)
    Y.append(row.mean_reporting_frequency)

X = np.array(X)
Y = np.array(Y)

model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")

pred = model.predict(X)

out = pd.DataFrame({
    "mean_reporting_frequency_true": Y,
    "mean_reporting_frequency_pred": pred,
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ Saved pred_fold{fold_id}.csv")

