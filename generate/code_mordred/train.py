import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# Helper: robust CSV loader with encoding fallback
# ======================================================
def robust_read_csv(path):
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            print(f"Trying to read with encoding: {enc}")
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            print(f"⚠ Failed with encoding {enc}, trying next...")
    raise ValueError("❌ All encoding attempts failed for: " + path)


# ======================================================
# 1. Load smell profile training data
# ======================================================
smell_df = pd.read_csv("train.csv")

# Fix problematic column names
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
# 3. Load Mordred Descriptors
# ======================================================
mordred_path = "../../sub1/data/raw/Mordred_Descriptors.csv"
mordred_df = robust_read_csv(mordred_path)

# Fix molecule dtype
mordred_df["molecule"] = mordred_df["molecule"].astype(str)

# Drop SMILES
if "SMILES" in mordred_df.columns:
    mordred_df.drop(columns=["SMILES"], inplace=True)

# Identify descriptor columns
descriptor_cols = [c for c in mordred_df.columns if c != "molecule"]

# Replace "" with NaN
mordred_df[descriptor_cols] = mordred_df[descriptor_cols].replace("", np.nan)

# Convert all descriptor columns to float
for c in descriptor_cols:
    mordred_df[c] = pd.to_numeric(mordred_df[c], errors="coerce")

print("Descriptor count:", len(descriptor_cols))


# ======================================================
# 4. Merge smell profile + mapping + descriptors
# ======================================================
df = smell_df.merge(stim2mol_df, on="stimulus", how="left")
df = df.merge(mordred_df, on="molecule", how="inner")

print("Final merged dataset shape:", df.shape)


# ======================================================
# 5. Build input X and target Y
# ======================================================
ignore_cols = {"stimulus", "molecule"}

smell_feature_cols = [
    c for c in df.columns
    if c not in ignore_cols and c not in descriptor_cols
]

X = df[smell_feature_cols]
Y = df[descriptor_cols]

print("Input X shape:", X.shape)
print("Target Y shape:", Y.shape)


# ======================================================
# 6. Remove constant descriptor columns
# ======================================================
n_unique = Y.nunique()
valid_descriptor_cols = n_unique[n_unique > 1].index.tolist()

print("Original Mordred dims:", len(descriptor_cols))
print("Non-constant dims:", len(valid_descriptor_cols))

if len(valid_descriptor_cols) == 0:
    raise ValueError("❌ All Mordred descriptors are constant!")

Y = Y[valid_descriptor_cols]

# Save descriptor list
pd.Series(valid_descriptor_cols).to_csv(
    "valid_mordred_cols.csv", index=False, header=False
)


# ======================================================
# 6b. Fill NaNs in Y (CatBoost does NOT allow NaN targets)
# ======================================================
print("Filling NaN values in Mordred targets (Y)...")
nan_before = Y.isna().sum().sum()

Y = Y.fillna(Y.mean())

nan_after = Y.isna().sum().sum()
print(f"✔ NaN fill complete. NaNs before: {nan_before}, after: {nan_after}")


# ======================================================
# 7. Train CatBoost MultiRMSE
# ======================================================
model = CatBoostRegressor(
    loss_function="MultiRMSE",
    depth=8,
    learning_rate=0.05,
    iterations=1000,
    random_seed=42,
    verbose=500
)

print("\nTraining model...")
model.fit(X, Y)


# ======================================================
# 8. Save model
# ======================================================
model.save_model("smell_to_mordred.json", format="json")

print("\n========================")
print("Training completed!")
print("Saved model: smell_to_mordred.json")
print("Saved descriptor list: valid_mordred_cols.csv")
print("========================\n")

