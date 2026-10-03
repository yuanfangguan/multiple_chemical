#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Predicting fold {fold_id}")

test_df = pd.read_csv(f"test_fold{fold_id}.csv")

maccs = pd.read_csv("../../data/features_maccs.csv")
maccs_map = dict(zip(
    maccs["SMILES"],
    maccs.drop(columns=["SMILES", "molecule_id"], errors="ignore").values
))

def maxpool(row):
    fp1 = maccs_map.get(row.drug_1_smiles)
    fp2 = maccs_map.get(row.drug_2_smiles)
    if fp1 is None or fp2 is None:
        return np.zeros(len(next(iter(maccs_map.values()))))
    return np.maximum(fp1, fp2)

X_test = np.vstack(test_df.apply(maxpool, axis=1).values)

model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")

pred = model.predict(X_test)

out = pd.DataFrame({
    "row_id": test_df["row_id"],
    "PRR_true": test_df["PRR"],
    "PRR_pred": pred
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ pred_fold{fold_id}.csv saved")

