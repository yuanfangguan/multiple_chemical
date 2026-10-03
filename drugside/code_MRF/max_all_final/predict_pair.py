#!/usr/bin/env python3
import pandas as pd
import numpy as np
from itertools import combinations
from catboost import CatBoostRegressor

# =======================
# 1. Load Model and Data
# =======================
print("Loading final model and training data...")
model = CatBoostRegressor()
model.load_model("final_catboost_model.cbm")

train_df = pd.read_csv("train_final.csv")

# =======================
# 2. Identify Novel Pairs
# =======================
# Get all unique drug SMILES from both columns
all_drugs = pd.concat([train_df['drug_1_smiles'], train_df['drug_2_smiles']]).unique()
all_drugs = [d for d in all_drugs if isinstance(d, str) and d != ""]

print(f"Total unique drugs found: {len(all_drugs)}")

# Create a set of existing pairs (order-independent) for fast lookup
existing_pairs = set()
for _, row in train_df.iterrows():
    pair = tuple(sorted([row['drug_1_smiles'], row['drug_2_smiles']]))
    existing_pairs.add(pair)

# Generate all possible combinations
print("Generating all possible novel combinations...")
all_possible_combos = list(combinations(all_drugs, 2))

novel_pairs = []
for d1, d2 in all_possible_combos:
    if tuple(sorted([d1, d2])) not in existing_pairs:
        novel_pairs.append({'drug_1_smiles': d1, 'drug_2_smiles': d2})

novel_df = pd.DataFrame(novel_pairs)
print(f"Found {len(novel_df)} novel pairs to predict.")

# =======================
# 3. Feature Loading & Fusion
# =======================
def load_fp(path, drop_cols=["SMILES", "molecule_id"]):
    df = pd.read_csv(path, low_memory=False)
    smiles = df["SMILES"]
    numeric = df.drop(columns=drop_cols, errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return dict(zip(smiles, numeric.values))

print("Loading feature maps...")
feats = {
    "morgan":  load_fp("../../data/features_morgan.csv"),
    "maccs":   load_fp("../../data/features_maccs.csv"),
    "rdkitfp": load_fp("../../data/features_rdkitfp.csv"),
    "desc":    load_fp("../../data/features_descriptors.csv"),
    "mordred": load_fp("../../data/features_mordred.csv")
}

def fuse_features(row):
    vectors = []
    for name, fp_map in feats.items():
        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None: return None
        vectors.append(np.maximum(fp1, fp2))
    return np.concatenate(vectors, axis=0)

# =======================
# 4. Predict
# =======================
print("Vectorizing and predicting...")
X_list = []
valid_indices = []

for idx, row in novel_df.iterrows():
    fused = fuse_features(row)
    if fused is not None:
        X_list.append(fused)
        valid_indices.append(idx)

if X_list:
    X_novel = np.array(X_list)
    predictions = model.predict(X_novel)
    
    # Map predictions back to the novel pairs
    final_results = novel_df.iloc[valid_indices].copy()
    final_results['predicted_mrf'] = predictions
    
    # Save results
    final_results.to_csv("novel_pair_predictions.csv", index=False)
    print("✔ Predictions saved to novel_pair_predictions.csv")
else:
    print("No valid features found for novel pairs.")
