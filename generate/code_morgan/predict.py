import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# 1. Load trained JSON model
# ======================================================
print("Loading model...")
# Assuming the model was saved as 'smell_to_morgan.json'
model = CatBoostRegressor()
model.load_model("smell_to_morgan.json", format="json")
print("Model loaded.\n")

# ======================================================
# 2. Load valid fingerprint column names - FIXED FOR INTEGER INDICES
# ======================================================
print("Loading valid fingerprint columns...")

try:
    # ⭐ FIX: Read the file, ensuring the index column (column 0) is loaded
    raw_indices_series = pd.read_csv("valid_fp_cols.csv", header=None, dtype=str)[0]
except FileNotFoundError:
    print("FATAL ERROR: valid_fp_cols.csv not found. Did training run successfully?")
    exit()

# Convert the column names (which are indices like '13', '16') to actual integers
valid_fp_indices = []
for index_str in raw_indices_series:
    try:
        # We only care about the integer value of the index
        valid_fp_indices.append(int(index_str))
    except ValueError:
        # This handles any potential header or corrupted non-integer value
        print(f"WARNING: Skipping non-integer index: {index_str}")

# Clean and sort the indices
valid_fp_indices = sorted(list(set(valid_fp_indices))) 

print("Parsed valid FP indices:", len(valid_fp_indices))
print()

# ======================================================
# 3. Load smell inputs and fix column names
# ======================================================
print("Loading smell test data...")
smell_df = pd.read_csv("test.csv")

# Fix: Rename columns to match the features the model was trained on
smell_df.rename(
    columns={
        "Garlic.Onion": "Garlic_Onion",
        "Rotten.Decay": "Rotten_Decay",
    },
    inplace=True
)

IGNORE_COLS = ["stimulus", "molecule", "SMILES"]
smell_feature_cols = [
    c for c in smell_df.columns 
    if c not in IGNORE_COLS and not c.startswith("bit_")
]

X_new = smell_df[smell_feature_cols]
print(f"Predicting on {X_new.shape[0]} test samples")
print(f"Smell feature dimension: {X_new.shape[1]}\n")

# ======================================================
# 4. Predict reduced FP
# ======================================================
print("Predicting reduced fingerprints...")
Y_pred_reduced = np.array(model.predict(X_new))
print("Reduced prediction shape:", Y_pred_reduced.shape)

n_model_outputs = Y_pred_reduced.shape[1]
print("Model outputs:", n_model_outputs)
print("FP indices stored:", len(valid_fp_indices))

# ======================================================
# 5. Fix mismatch and align prediction
# ======================================================
if n_model_outputs != len(valid_fp_indices):
    print("\n⚠ WARNING: MISMATCH detected between model outputs and valid_fp_cols.csv")
    print(f" -> Model predicts {n_model_outputs} FP bits")
    print(f" -> valid_fp_cols.csv contains {len(valid_fp_indices)} bits")

    # Align the indices to the number of outputs the model actually produced
    # ⭐ IMPORTANT: This assumes the indices in the file are in the same order 
    # as the model's output targets.
    valid_fp_indices = valid_fp_indices[:n_model_outputs]

    print(f"Fixed FP index count: {len(valid_fp_indices)} (should now match model output)\n")

# ======================================================
# 6. Reconstruct full 2048-dim FP (MACCS dimension)
# ======================================================
FULL_DIM = 2048 # MACCS dimension
N = X_new.shape[0]

print(f"Reconstructing full {FULL_DIM}-bit fingerprints...")
full_fp_array = np.zeros((N, FULL_DIM), dtype=float)

# Map the predicted values back to their correct positions (indices)
# This requires Y_pred_reduced to have the exact same number of columns as valid_fp_indices
full_fp_array[:, valid_fp_indices] = Y_pred_reduced

print("Full FP shape:", full_fp_array.shape)

# ======================================================
# 7. Save outputs
# ======================================================
print("\nSaving outputs...")
np.save("predicted_full_fingerprints.npy", full_fp_array)

# ⭐ CRITICAL FIX: Ensure the column names match the full dimension (2048)
fp_df = pd.DataFrame(full_fp_array, columns=[f"bit_{i}" for i in range(2048)])
fp_df.to_csv("predicted_full_fingerprints.csv", index=False)

smell_df.to_csv("predicted_fp_metadata.csv", index=False)

print("\n==============================")
print("Prediction completed SUCCESSFULLY!")
print("Saved: predicted_full_fingerprints.csv")
print("==============================")
