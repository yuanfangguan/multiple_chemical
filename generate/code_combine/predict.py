import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import os

# ======================================================
# 1. Configuration & Setup
# ======================================================
MODEL_FILE = "smell_to_all_features.json"
COLS_FILE = "valid_all_features_cols.csv"
TEST_FILE = "test.csv"

# Define reconstruction parameters for the different feature types
# dim: The full size of the fingerprint vector (for reconstruction)
# type: 'real' (direct values) or 'fp' (reconstruct sparse bit vector)
FEATURE_CONFIG = {
    "desc":    {"type": "real", "outfile": "predicted_descriptors.csv"},
    "mordred": {"type": "real", "outfile": "predicted_mordred_descriptors.csv"},
    "maccs":   {"type": "fp",   "dim": 167,  "outfile": "predicted_maccs.csv"},   # RDKit MACCS is typically 167 bits
    "morgan":  {"type": "fp",   "dim": 2048, "outfile": "predicted_morgan.csv"},
    "rdkit":   {"type": "fp",   "dim": 2048, "outfile": "predicted_rdkit.csv"}
}

# ======================================================
# 2. Load Resources
# ======================================================
print(f"Loading valid column definitions from {COLS_FILE}...")
try:
    # Load the list of columns the model was trained on (e.g., 'maccs_bit_1', 'desc_MolWt')
    valid_cols = pd.read_csv(COLS_FILE, header=None)[0].tolist()
except FileNotFoundError:
    print(f"FATAL ERROR: {COLS_FILE} not found. Run training first.")
    exit()

print(f"Loading model from {MODEL_FILE}...")
model = CatBoostRegressor()
model.load_model(MODEL_FILE, format="json")

# ======================================================
# 3. Load and Preprocess Test Data
# ======================================================
print("Loading smell test data...")
smell_df = pd.read_csv(TEST_FILE)

# Standardize column names (same as training)
smell_df.rename(
    columns={
        "Garlic.Onion": "Garlic_Onion",
        "Rotten.Decay": "Rotten_Decay",
    },
    inplace=True
)

# Prepare Input Matrix X
# We use the same exclusion logic as training to isolate smell features
IGNORE_COLS = ["stimulus", "molecule", "SMILES"]
smell_feature_cols = [
    c for c in smell_df.columns 
    if c not in IGNORE_COLS and not any(c.startswith(k) for k in ["bit_", "desc_", "maccs_", "morgan_", "rdkit_", "mordred_"])
]

X_new = smell_df[smell_feature_cols]
print(f"Predicting on {X_new.shape[0]} samples with {X_new.shape[1]} input features.\n")

# ======================================================
# 4. Global Prediction
# ======================================================
print("Running global prediction...")
# This returns a matrix of shape (N_samples, N_valid_cols)
Y_pred_all = model.predict(X_new)

# Convert to DataFrame for easy column slicing based on names
Y_pred_df = pd.DataFrame(Y_pred_all, columns=valid_cols)
print("Global prediction complete. Separating features...\n")

# ======================================================
# 5. Separation and Reconstruction
# ======================================================
N_samples = len(smell_df)

for prefix, config in FEATURE_CONFIG.items():
    print(f"--- Processing {prefix.upper()} ---")
    
    # 5a. Identify columns belonging to this feature set
    # The training script prefixed columns like 'maccs_bit_12' or 'desc_MolWt'
    relevant_cols = [c for c in valid_cols if c.startswith(f"{prefix}_")]
    
    if not relevant_cols:
        print(f"Warning: No columns found for {prefix}. Skipping.")
        continue
        
    print(f"Found {len(relevant_cols)} predicted features.")
    
    # Extract the subset of predictions
    subset_df = Y_pred_df[relevant_cols]
    
    # 5b. Handle Real-valued Descriptors (Desc, Mordred)
    if config["type"] == "real":
        # Remove the prefix to restore original names (e.g., 'desc_MolWt' -> 'MolWt')
        clean_names = [c.replace(f"{prefix}_", "") for c in relevant_cols]
        subset_df.columns = clean_names
        
        # Save
        subset_df.to_csv(config["outfile"], index=False)
        print(f"Saved {config['outfile']}")

    # 5c. Handle Fingerprints (Reconstruction required)
    elif config["type"] == "fp":
        dim = config["dim"]
        
        # Initialize full matrix with zeros
        full_fp_array = np.zeros((N_samples, dim), dtype=float)
        
        # Map predicted columns to their integer indices
        # Example: 'maccs_bit_42' -> index 42
        mapped_count = 0
        
        for col in relevant_cols:
            # Strip prefix first: 'maccs_bit_42' -> 'bit_42'
            original_name = col.replace(f"{prefix}_", "")
            
            # Extract Integer Index
            try:
                if "bit_" in original_name:
                    idx = int(original_name.split("_")[-1])
                else:
                    idx = int(original_name)
                
                # Assign values if index is within bounds
                if 0 <= idx < dim:
                    full_fp_array[:, idx] = subset_df[col].values
                    mapped_count += 1
            except ValueError:
                print(f"Skipping unparseable column index: {col}")

        # Save as CSV with bit_X headers
        fp_cols = [f"bit_{i}" for i in range(dim)]
        out_df = pd.DataFrame(full_fp_array, columns=fp_cols)
        out_df.to_csv(config["outfile"], index=False)
        
        # Save numpy array for advanced usage
        npy_filename = config["outfile"].replace(".csv", ".npy")
        np.save(npy_filename, full_fp_array)
        
        print(f"Reconstructed {dim}-bit vector (Mapped {mapped_count} features).")
        print(f"Saved {config['outfile']}")

# ======================================================
# 6. Save Metadata
# ======================================================
smell_df.to_csv("predicted_metadata.csv", index=False)
print("\n==============================")
print("All predictions separated and saved successfully!")
print("Metadata saved to: predicted_metadata.csv")
print("==============================")
