#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

fold_id = int(sys.argv[1])
pred_df = pd.read_csv(f"pred_fold{fold_id}.csv")

y_true = pred_df["PRR_true"].values
y_pred = pred_df["PRR_pred"].values

corr, _ = pearsonr(y_true, y_pred)
cos_sim = 1 - cosine(y_true, y_pred)

with open(f"metrics_fold{fold_id}.txt", "w") as f:
    f.write(f"Pearson correlation: {corr}\n")
    f.write(f"Cosine similarity: {cos_sim}\n")

print(f"✔ metrics_fold{fold_id}.txt saved")

