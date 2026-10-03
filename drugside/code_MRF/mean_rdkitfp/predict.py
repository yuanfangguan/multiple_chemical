#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Predicting fold {fold_id}")

test_df = pd.read_csv(f"test_fold{fold_id}.csv")

rdkitfp = pd.read_csv("../../data/features_rdkitfp.csv")
rdkitfp_map = dict(zip(
    rdkitfp["SMILES"],
    rdkitfp.drop(columns=["SMILES", "molecule_id"], errors="ignore").values
))

def meanpool(row):
    fp1 = rdkitfp_map.get(row.drug_1_smiles)
    fp2 = rdkitfp_map.get(row.drug_2_smiles)
    if fp1 is None or fp2 is None:
        return np.zeros(len(next(iter(rdkitfp_map.values()))))
    return np.mean((fp1, fp2),axis=0)

X_test = np.vstack(test_df.apply(meanpool, axis=1).values)

model = CatBoostRegressor()
model.load_model(f"model_fold{fold_id}.cbm")

pred = model.predict(X_test)

out = pd.DataFrame({
    "row_id": test_df["row_id"],
    "mean_reporting_frequency_true": test_df["mean_reporting_frequency"],
    "mean_reporting_frequency_pred": pred
})

out.to_csv(f"pred_fold{fold_id}.csv", index=False)
print(f"✔ pred_fold{fold_id}.csv saved")

