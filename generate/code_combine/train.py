import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# Helper: robust CSV loader (Required for Mordred file)
# ======================================================
def robust_read_csv(path):
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            print(f"Loading {path} with encoding: {enc}")
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"❌ Failed to read {path} with common encodings.")

# ======================================================
# 1. Load smell profile training data (The Input X)
# ======================================================
print("--- Loading Smell Data ---")
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

# Initialize main dataframe with smell data + molecule mapping
main_df = smell_df.merge(stim2mol_df, on="stimulus", how="left")
print(f"Base data shape: {main_df.shape}")

# ======================================================
# 3. Load and Concatenate All Feature Sets
# ======================================================
# Dictionary mapping prefix name -> file path
feature_files = {
    "desc": "../../sub1/data/processed/features_descriptors.csv",
    "maccs": "../../sub1/data/processed/features_maccs.csv",
    "morgan": "../../sub1/data/processed/features_morgan.csv",
    "rdkit": "../../sub1/data/processed/features_rdkitfp.csv",
    "mordred": "../../sub1/data/raw/Mordred_Descriptors.csv"
}

for prefix, path in feature_files.items():
    print(f"\n--- Processing {prefix.upper()} features ---")
    
    # 3a. Load Data
    if prefix == "mordred":
        feat_df = robust_read_csv(path)
    else:
        feat_df = pd.read_csv(path)
    
    # 3b. Basic Cleaning
    feat_df["molecule"] = feat_df["molecule"].astype(str)
    
    if "SMILES" in feat_df.columns:
        feat_df.drop(columns=["SMILES"], inplace=True)
        
    # 3c. Specific Cleaning for Mordred (non-numeric handling)
    if prefix == "mordred":
        # Identify non-molecule columns
        m_cols = [c for c in feat_df.columns if c != "molecule"]
        # Coerce to numeric (handles strings like 'Error' or empty strings)
        for c in m_cols:
             feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    
    # 3d. Rename columns with prefix to avoid collisions 
    # (e.g., bit_1 in MACCS vs bit_1 in Morgan)
    rename_map = {c: f"{prefix}_{c}" for c in feat_df.columns if c != "molecule"}
    feat_df.rename(columns=rename_map, inplace=True)
    
    # 3e. Merge into main dataframe
    # We use inner merge to ensure we only keep molecules we have features for
    prev_shape = main_df.shape
    main_df = main_df.merge(feat_df, on="molecule", how="inner")
    
    print(f"Merged {prefix}. New shape: {main_df.shape}")
    print(f"Added {main_df.shape[1] - prev_shape[1]} feature columns.")

# ======================================================
# 4. Build Input (X) and Target (Y)
# ======================================================
ignore_cols = {"stimulus", "molecule"}

# Identify Smell columns (X) - assumed to be existing columns in smell_df
# We look at the original smell_df columns to identify X
original_smell_cols = [c for c in smell_df.columns if c != "stimulus"]
smell_feature_cols = [c for c in main_df.columns if c in original_smell_cols]

# Identify Target columns (Y) - everything else
feature_cols = [
    c for c in main_df.columns 
    if c not in ignore_cols and c not in smell_feature_cols
]

X = main_df[smell_feature_cols]
Y = main_df[feature_cols]

print("\n--- Matrix Construction ---")
print("Input (X) shape:", X.shape)
print("Target (Y) shape:", Y.shape)

# ======================================================
# 5. Data Cleaning: Constants & NaNs
# ======================================================

# 5a. Remove Constant Columns
n_unique = Y.nunique()
valid_cols = n_unique[n_unique > 1].index.tolist()

print(f"Original feature count: {len(feature_cols)}")
print(f"Non-constant feature count: {len(valid_cols)}")

if len(valid_cols) == 0:
    raise ValueError("All feature columns are constant!")

Y = Y[valid_cols]

# 5b. Handle NaNs (Critical for Mordred/Merged data)
# CatBoost MultiRMSE cannot handle NaNs in target
nan_count = Y.isna().sum().sum()
if nan_count > 0:
    print(f"Found {nan_count} NaN values in targets. Filling with column means...")
    Y = Y.fillna(Y.mean())
    # Final check just in case a column was ALL NaNs
    Y = Y.fillna(0) 

# Save valid list
pd.Series(valid_cols).to_csv("valid_all_features_cols.csv", index=False, header=False)

# ======================================================
# 6. Train CatBoost MultiRMSE model
# ======================================================
print("\n--- Training CatBoost Model ---")
model = CatBoostRegressor(
    loss_function="MultiRMSE",
    depth=8,
    learning_rate=0.05,
    iterations=2000, # Increased slightly as problem is now harder/wider
    random_seed=42,
    verbose=500,
    task_type="CPU" 
)

model.fit(X, Y)

# ======================================================
# 7. Save model
# ======================================================
model.save_model("smell_to_all_features.json", format="json")

print("\n========================")
print("Training completed!")
print("Saved model: smell_to_all_features.json")
print("Saved columns: valid_all_features_cols.csv")
print("========================\n")
