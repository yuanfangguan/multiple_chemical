import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# 1. Load trained JSON model
# ======================================================
print("Loading model...")
# Assuming the model was saved as 'smell_to_descriptor.json'
model = CatBoostRegressor()
model.load_model("smell_to_descriptors.json", format="json")
print("Model loaded.\n")

# ======================================================
# 2. Load valid descriptor column names (These are strings like MolWt)
# ======================================================
print("Loading valid descriptor columns...")
# Load the 5 descriptor names (MolWt, HDonors, etc.)
# We do NOT try to convert these strings to indices.
try:
    raw_cols = pd.read_csv("valid_fp_cols.csv", header=None)[0].tolist() 
except FileNotFoundError:
    print("FATAL ERROR: valid_fp_cols.csv not found. Did training run successfully?")
    exit()

descriptor_cols = [c for c in raw_cols if isinstance(c, str)]

print("Valid Descriptor columns in file:", len(descriptor_cols))
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
# 4. Predict output (Descriptors)
# ======================================================
print("Predicting descriptors...")
Y_pred_descriptors = np.array(model.predict(X_new))
print("Predicted descriptors shape:", Y_pred_descriptors.shape)

n_model_outputs = Y_pred_descriptors.shape[1]
print("Model outputs:", n_model_outputs)
print("Descriptor features stored:", len(descriptor_cols))

# ======================================================
# 5. Save outputs (Descriptors ARE the final output)
# ⭐ NO RECONSTRUCTION/INDEXING STEP IS NEEDED HERE ⭐
# ======================================================
predicted_output_array = Y_pred_descriptors
N = X_new.shape[0]

# Sanity check: the number of predicted features must match the number of column names
if predicted_output_array.shape[1] != len(descriptor_cols):
    raise ValueError(
        f"Prediction shape mismatch! Model output {predicted_output_array.shape[1]} "
        f"does not match loaded column names {len(descriptor_cols)}."
    )

print("\nSaving outputs...")

# Save the predicted descriptor values using the descriptor column names
predicted_descriptors_df = pd.DataFrame(
    predicted_output_array, 
    columns=descriptor_cols # Use the names: MolWt, HDonors, etc.
)

# Use a new filename that reflects the content (descriptors)
predicted_descriptors_df.to_csv("predicted_descriptors.csv", index=False) 

# Save metadata (test data)
smell_df.to_csv("predicted_fp_metadata.csv", index=False) 

print("\n==============================")
print("Prediction completed SUCCESSFULLY!")
print("Saved: predicted_descriptors.csv")
print("==============================")
