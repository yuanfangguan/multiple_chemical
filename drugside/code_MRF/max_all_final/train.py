#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

# Use a fixed identifier for the final model
print("Training Final Production Model...")

# =======================
# Load Full Dataset
# =======================
# Using the train_final.csv we generated in the previous step
train_df = pd.read_csv("train_final.csv")

# =======================
# Clean mean_reporting_frequency & SMILES
# =======================
train_df["mean_reporting_frequency"] = pd.to_numeric(train_df["mean_reporting_frequency"], errors="coerce")
train_df = train_df.dropna(subset=["mean_reporting_frequency", "drug_1_smiles", "drug_2_smiles"])
train_df = train_df[(train_df["drug_1_smiles"] != "") & (train_df["drug_2_smiles"] != "")]

# =======================
# Helper: load FP dataset
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
print("✔ feature maps ready")

# =======================
# Unified pooling (Max Pooling)
# =======================
def fuse_features(row):
    vectors = []
    for name, fp_map in feats.items():
        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            return None
        pooled = np.maximum(fp1, fp2)
        vectors.append(pooled)
    return np.concatenate(vectors, axis=0)

# =======================
# Convert DF into matrix
# =======================
def df_to_matrix(df):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        fused = fuse_features(row)
        if fused is None:
            continue
        X_list.append(fused)
        y_list.append(row.mean_reporting_frequency)
    return np.array(X_list), np.array(y_list)

X_train, y_train = df_to_matrix(train_df)

print(f"Final feature vector size = {X_train.shape[1]}")
print(f"Total training samples = {X_train.shape[0]}")

# =======================
# train CatBoost (Full Data)
# =======================
# Note: eval_set is removed as we are using 100% of data for training
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=42, # Static seed for reproducibility
    verbose=200,
)

# Fit on all available data
model.fit(X_train, y_train)

# Save the final model
model.save_model("final_catboost_model.cbm")
print("✔ Saved final_catboost_model.cbm")
