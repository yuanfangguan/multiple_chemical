#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import KFold
import sys

seed = int(sys.argv[1])
print(f"Using seed = {seed}")

df = pd.read_csv("../../data/TWOSIDES_drugpair_clean.csv")
df["row_id"] = df.index

kf = KFold(n_splits=5, shuffle=True, random_state=seed)

for fold_id, (train_idx, test_idx) in enumerate(kf.split(df)):
    df.loc[train_idx].to_csv(f"train_fold{fold_id}.csv", index=False)
    df.loc[test_idx].to_csv(f"test_fold{fold_id}.csv", index=False)

print("✔ Generated train/test CSV for 5 folds.")

