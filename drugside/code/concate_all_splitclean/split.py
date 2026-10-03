#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import KFold

seed = int(sys.argv[1])
print(f"Using seed =", seed)

# =======================
# Load data
# =======================
df = pd.read_csv("../../data/TWOSIDES_drugpair_clean.csv")
df["row_id"] = df.index

# all unique drugs
all_drugs = pd.unique(
    pd.concat([df["drug_1_smiles"], df["drug_2_smiles"]], axis=0)
)

rng = np.random.default_rng(seed)

# =======================
# 5 bipartite folds
# =======================
for fold_id in range(5):

    # ------------
    # split drugs
    # ------------
    rng.shuffle(all_drugs)
    n = int(len(all_drugs) * 0.70)   # target proportion

    train_drugs = set(all_drugs[:n])
    test_drugs  = set(all_drugs[n:])

    # ------------
    # assign pairs to splits
    # ------------
    train_pairs = []
    test_pairs  = []
    skip_pairs  = []  # OPTIONAL

    for _, row in df.iterrows():
        d1 = row.drug_1_smiles
        d2 = row.drug_2_smiles

        if d1 in train_drugs and d2 in train_drugs:
            train_pairs.append(row)
        elif d1 in test_drugs and d2 in test_drugs:
            test_pairs.append(row)
        else:
            # cross pair
            skip_pairs.append(row)

    # convert back to DataFrame
    train_df = pd.DataFrame(train_pairs)
    test_df  = pd.DataFrame(test_pairs)

    print(f"\nFold {fold_id}")
    print("Train drugs =", len(train_drugs))
    print("Test drugs  =", len(test_drugs))
    print("Train pairs =", len(train_df))
    print("Test pairs  =", len(test_df))
    print("Skipped pairs (cross) =", len(skip_pairs))

    # =======================
    # save CSV
    # =======================
    train_df.to_csv(f"train_fold{fold_id}.csv", index=False)
    test_df.to_csv(f"test_fold{fold_id}.csv", index=False)

print("\n✔ Finished bipartite splits.")

