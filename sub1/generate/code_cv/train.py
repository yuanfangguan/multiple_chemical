import pandas as pd
from catboost import CatBoostRegressor

# ======================================================
# 1. Load smell profile training data
# ======================================================
smell_df = pd.read_csv("train.csv")   # stimulus + ~55 smell features

# ======================================================
# 2. Load stimulus → molecule mapping
# ======================================================
stim2mol_df = pd.read_csv("../../data/raw/TASK1_Stimulus_definition.csv")
stim2mol_df = stim2mol_df[["stimulus", "molecule"]]

# ======================================================
# 3. Load fingerprint data (correct path)
# ======================================================
fp_df = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")
# columns: molecule, SMILES, bit_0 … bit_2047

# ======================================================
# 4. Merge smell profile + molecule mapping
# ======================================================
df = smell_df.merge(stim2mol_df, on="stimulus", how="left")

# Now merge fingerprints
df = df.merge(fp_df, on="molecule", how="inner")

print("Final merged dataset shape:", df.shape)

# ======================================================
# 5. Build input matrix X and target matrix Y
# ======================================================
ignore_cols = ["stimulus", "molecule", "SMILES"]

# smell feature columns (inputs)
smell_feature_cols = [
    c for c in df.columns
    if c not in ignore_cols and not c.startswith("bit_")
]

# fingerprint columns (targets)
fingerprint_cols = [c for c in df.columns if c.startswith("bit_")]

X = df[smell_feature_cols]
Y = df[fingerprint_cols]

print("Input (X) shape:", X.shape)
print("Target (Y) shape:", Y.shape)

# ======================================================
# 6. Remove constant fingerprint columns
# ======================================================
n_unique = Y.nunique()
valid_fp_cols = n_unique[n_unique > 1].index.tolist()

print("Original FP dims:", len(fingerprint_cols))
print("Non-constant FP dims:", len(valid_fp_cols))

Y = Y[valid_fp_cols]

# Save list of columns we keep
pd.Series(valid_fp_cols).to_csv("valid_fp_cols.csv", index=False)

# ======================================================
# 7. Train CatBoost MultiRMSE model
# ======================================================
model = CatBoostRegressor(
    loss_function="MultiRMSE",
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    random_seed=42,
    verbose=200
)

print("Training model...")
model.fit(X, Y)

# ======================================================
# 8. Save model in JSON format (bulletproof)
# ======================================================
model.save_model("smell_to_fingerprint.json", format="json")

print("\n========================")
print("Training completed!")
print("Saved model: smell_to_fingerprint.json")
print("Saved valid FP index list: valid_fp_cols.csv")
print("========================\n")

