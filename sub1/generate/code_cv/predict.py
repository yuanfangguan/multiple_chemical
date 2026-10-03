import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# 1. Load trained JSON model
# ======================================================
model = CatBoostRegressor()
model.load_model("smell_to_fingerprint.json", format="json")

# ======================================================
# 2. Load valid fingerprint column names
# ======================================================
raw_cols = pd.read_csv("valid_fp_cols.csv", header=None)[0].tolist()

# Keep only columns that look like bit_XXX
valid_fp_cols = [c for c in raw_cols if isinstance(c, str) and c.startswith("bit_")]

print("Reduced fingerprint dims:", len(valid_fp_cols))

# Convert bit_* names → integer indices
valid_fp_indices = []
for col in valid_fp_cols:
    try:
        idx = int(col.split("_")[1])
        valid_fp_indices.append(idx)
    except Exception:
        print("WARNING: skipping malformed column:", col)

# Final check
print("Usable fingerprint indices:", len(valid_fp_indices))

# ======================================================
# 3. Load smell input
# ======================================================
smell_df = pd.read_csv("test.csv")

ignore_cols = ["stimulus", "molecule", "SMILES"]
smell_feature_cols = [
    c for c in smell_df.columns
    if c not in ignore_cols and not c.startswith("bit_")
]

X_new = smell_df[smell_feature_cols]

print("Predicting on", X_new.shape[0], "samples")

# ======================================================
# 4. Predict reduced FP
# ======================================================
Y_pred_reduced = np.array(model.predict(X_new))
print("Reduced prediction shape:", Y_pred_reduced.shape)

# ======================================================
# 5. Reconstruct full 2048-dim FP
# ======================================================
N = Y_pred_reduced.shape[0]
FULL_DIM = 2048

full_fp_array = np.zeros((N, FULL_DIM), dtype=float)

for i in range(N):
    for j, bit_index in enumerate(valid_fp_indices):
        full_fp_array[i, bit_index] = Y_pred_reduced[i, j]

print("Full FP shape:", full_fp_array.shape)

# ======================================================
# 6. Save
# ======================================================
np.save("predicted_full_fingerprints.npy", full_fp_array)
fp_df = pd.DataFrame(full_fp_array, columns=[f"bit_{i}" for i in range(2048)])
fp_df.to_csv("predicted_full_fingerprints.csv", index=False)
smell_df.to_csv("predicted_fp_metadata.csv", index=False)

print("Done.")

