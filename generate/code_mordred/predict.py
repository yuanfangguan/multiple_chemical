import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ======================================================
# 1. Load trained Mordred model
# ======================================================
print("Loading model...")

model = CatBoostRegressor()
model.load_model("smell_to_mordred.json", format="json")

print("Model loaded.\n")


# ======================================================
# 2. Load valid Mordred descriptor column names
# ======================================================
print("Loading valid Mordred descriptor names...")

try:
    valid_cols = pd.read_csv("valid_mordred_cols.csv", header=None)[0].tolist()
except FileNotFoundError:
    print("FATAL ERROR: valid_mordred_cols.csv not found.")
    exit()

print(f"Loaded {len(valid_cols)} Mordred descriptor names.\n")


# ======================================================
# 3. Load smell test input data
# ======================================================
print("Loading smell test data...")

smell_df = pd.read_csv("test.csv")

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
    if c not in IGNORE_COLS
]

X_new = smell_df[smell_feature_cols]

print(f"Predicting on {X_new.shape[0]} samples")
print(f"Input smell feature dimension: {X_new.shape[1]}\n")


# ======================================================
# 4. Predict Mordred descriptors
# ======================================================
print("Predicting Mordred descriptors...")

Y_pred = np.array(model.predict(X_new))

print("Prediction shape:", Y_pred.shape)
print(f"Expected descriptor dimension: {len(valid_cols)}")

if Y_pred.shape[1] != len(valid_cols):
    print("\n⚠ ERROR: Output dimension mismatch.")
    print("Model output:", Y_pred.shape[1], " / Expected:", len(valid_cols))
    exit()

print()


# ======================================================
# 5. Save outputs
# ======================================================
print("Saving predictions...")

np.save("predicted_mordred_descriptors.npy", Y_pred)

pred_df = pd.DataFrame(Y_pred, columns=valid_cols)
pred_df.to_csv("predicted_mordred_descriptors.csv", index=False)

# Save metadata (stimulus, molecule, other smell fields)
smell_df.to_csv("predicted_mordred_metadata.csv", index=False)

print("\n==============================")
print("Prediction COMPLETED SUCCESSFULLY!")
print("Saved: predicted_mordred_descriptors.csv")
print("==============================\n")

