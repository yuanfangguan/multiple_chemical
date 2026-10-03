#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

# =======================
# Args
# =======================
fold_id = int(sys.argv[1])
print(f"Training fold {fold_id}")

# =======================
# Load train / test
# =======================
train_df = pd.read_csv(f"train_fold{fold_id}.csv")
test_df  = pd.read_csv(f"test_fold{fold_id}.csv")

# =======================
# Clean PRR & SMILES
# =======================
for df in (train_df, test_df):
    df["PRR"] = pd.to_numeric(df["PRR"], errors="coerce")
    df.dropna(subset=["PRR","drug_1_smiles","drug_2_smiles"], inplace=True)
    df = df[(df["drug_1_smiles"]!="") & (df["drug_2_smiles"]!="")]

# =======================
# Helper: load FP dataset and normalize numeric
# =======================
def load_fp(path, drop_cols=["SMILES","molecule_id"]):
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
    "mordred": load_fp("../../data/features_mordred.csv"),
}

print("✔ feature maps ready")

# =======================
# unified single-drug feature extraction
# =======================
def get_features_single(smiles):
    vectors = []
    for _, fp_map in feats.items():
        fp = fp_map.get(smiles)
        if fp is None:
            return None
        vectors.append(fp)
    return np.concatenate(vectors, axis=0)

# =======================
# convert DF → ONE example per pair (concatenate drug1 + drug2)
# =======================
def df_to_matrix(df):
    X_list, y_list = [], []

    for _, row in df.iterrows():
        f1 = get_features_single(row.drug_1_smiles)
        f2 = get_features_single(row.drug_2_smiles)

        # skip if either FP is missing
        if f1 is None or f2 is None:
            continue

        # 🔥 NEW: single fused vector
        fused = np.concatenate([f1, f2], axis=0)

        X_list.append(fused)
        y_list.append(row.PRR)

    return np.array(X_list), np.array(y_list)

# =======================
# Build matrices
# =======================
X_train, y_train = df_to_matrix(train_df)
X_test,  y_test  = df_to_matrix(test_df)

print("Final feature vector size =", X_train.shape[1])
print("Train size =", X_train.shape)
print("Test size  =", X_test.shape)

# =======================
# Train CatBoost
# =======================
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=fold_id,
    verbose=200,
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# =======================
# Save model
# =======================
model.save_model(f"model_fold{fold_id}.cbm")
print(f"✔ Saved model_fold{fold_id}.cbm")

