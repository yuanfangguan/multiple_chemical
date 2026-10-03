#!/usr/bin/env python3
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# =======================
# 1. Load Model and Training Data
# =======================
print("Loading model and training pairs...")
model = CatBoostRegressor()
model.load_model("final_catboost_model.cbm")

# Load existing pairs
train_df = pd.read_csv("train_final.csv")
# Get unique drug library from the training set
all_drugs = pd.concat([train_df['drug_1_smiles'], train_df['drug_2_smiles']]).unique()
all_drugs = [d for d in all_drugs if isinstance(d, str) and d != ""]

print(f"Unique drugs available to add: {len(all_drugs)}")

# =======================
# 2. Feature Loading
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

# =======================
# 3. Triple Fusion Function
# =======================
def fuse_triple_features(s1, s2, s3):
    """
    Applies Max-Pooling across THREE drug vectors for each feature type.
    """
    vectors = []
    for name, fp_map in feats.items():
        v1 = fp_map.get(s1)
        v2 = fp_map.get(s2)
        v3 = fp_map.get(s3)
        
        if v1 is None or v2 is None or v3 is None:
            return None
            
        # Element-wise max across all three drugs
        pooled = np.maximum(np.maximum(v1, v2), v3)
        vectors.append(pooled)
    return np.concatenate(vectors, axis=0)

# =======================
# 4. Generate and Predict Triples
# =======================
print("Generating triples and predicting...")

triple_results = []

# To keep this computationally feasible, we'll process 
# a sample or specific subset if the list is huge.
# Here we iterate through existing pairs and add a 3rd drug.
for idx, row in train_df.iterrows():
    s1, s2 = row['drug_1_smiles'], row['drug_2_smiles']
    
    for s3 in all_drugs:
        # Avoid adding a drug that is already in the pair
        if s3 == s1 or s3 == s2:
            continue
            
        fused = fuse_triple_features(s1, s2, s3)
        if fused is not None:
            pred = model.predict(fused)
            triple_results.append({
                'base_drug_1': s1,
                'base_drug_2': s2,
                'added_drug_3': s3,
                'predicted_mrf': pred
            })
    
    # Optional: limit for testing to avoid massive file sizes
    if idx == 100: 
        print("Reached 100 base pairs limit for demonstration.")
        break

# Save results
out_df = pd.DataFrame(triple_results)
out_df.to_csv("triple_drug_predictions.csv", index=False)
print(f"✔ Saved {len(out_df)} triple-drug predictions to triple_drug_predictions.csv")
