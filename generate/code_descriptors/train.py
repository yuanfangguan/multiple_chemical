import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# 1. Load smell profile training data & Rename columns
# ======================================================
smell_df = pd.read_csv("train.csv") 

# Fix columns with '.' for consistency
smell_df.rename(
    columns={
        "Garlic.Onion": "Garlic_Onion",
        "Rotten.Decay": "Rotten_Decay",
    },
    inplace=True
)

# ======================================================
# 2. Load stimulus → molecule mapping
# ======================================================
stim2mol_df = pd.read_csv("../../sub1/data/raw/TASK1_Stimulus_definition.csv")
stim2mol_df = stim2mol_df[["stimulus", "molecule"]]
stim2mol_df["molecule"] = stim2mol_df["molecule"].astype(str)

# ======================================================
# 3. Load the MACCS fingerprint file (Must contain 166 bits)
# ======================================================
fp_df = pd.read_csv("../../sub1/data/processed/features_descriptors.csv")
fp_df["molecule"] = fp_df["molecule"].astype(str)

# Ensure FP column names are strings starting with 'bit_' 
# MACCS FP indices typically run from 1 to 166, but we'll use 0-indexing for simplicity if needed.
fp_cols = [c for c in fp_df.columns if c != "molecule" and c != "SMILES"]

print("MACCS fingerprint columns detected:", len(fp_cols))

# ======================================================
# 4. Merge smell profile + molecule mapping + fingerprints
# ======================================================
df = smell_df.merge(stim2mol_df, on="stimulus", how="left")
df = df.merge(fp_df, on="molecule", how="inner")

print("Final merged dataset shape:", df.shape)

# ======================================================
# 5. Build input matrix X and target matrix Y
# ======================================================
ignore_cols = {"stimulus", "molecule"}

smell_feature_cols = [
    c for c in df.columns 
    if c not in ignore_cols and c not in fp_cols
]

fingerprint_cols = fp_cols

X = df[smell_feature_cols]
Y = df[fingerprint_cols]

print("Input (X) shape:", X.shape)
print("Target (Y) shape:", Y.shape)

# ======================================================
# 6. Remove constant fingerprint columns and save names
# ======================================================
n_unique = Y.nunique()
valid_fp_cols = n_unique[n_unique > 1].index.tolist()

print("Original FP dims (MACCS):", len(fingerprint_cols))
print("Non-constant FP dims:", len(valid_fp_cols))

if len(valid_fp_cols) == 0:
    raise ValueError("All MACCS fingerprint columns are constant. Check data diversity!")

Y = Y[valid_fp_cols]

# Save valid list: These are the columns the model will predict.
pd.Series(valid_fp_cols).to_csv("valid_fp_cols.csv", index=False, header=False)

# ======================================================
# 7. Train CatBoost MultiRMSE model
# ======================================================
model = CatBoostRegressor(
    loss_function="MultiRMSE",
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    random_seed=42,
    verbose=500
)

print("Training model...")
model.fit(X, Y)

# ======================================================
# 8. Save model
# ======================================================
model.save_model("smell_to_descriptors.json", format="json")

print("\n========================")
print("Training completed!")
print("Saved valid FP list: valid_fp_cols.csv")
print("========================\n")
