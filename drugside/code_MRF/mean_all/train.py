#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
from catboost import CatBoostRegressor

fold_id = int(sys.argv[1])
print(f"Training fold {fold_id}")

# =======================
# Load train / test
# =======================
train_df = pd.read_csv(f"train_fold{fold_id}.csv")
test_df  = pd.read_csv(f"test_fold{fold_id}.csv")

# =======================
# Clean mean_reporting_frequency & SMILES
# =======================
train_df["mean_reporting_frequency"] = pd.to_numeric(train_df["mean_reporting_frequency"], errors="coerce")
test_df ["mean_reporting_frequency"] = pd.to_numeric(test_df ["mean_reporting_frequency"], errors="coerce")

train_df = train_df.dropna(subset=["mean_reporting_frequency","drug_1_smiles","drug_2_smiles"])
test_df  = test_df.dropna(subset=["mean_reporting_frequency","drug_1_smiles","drug_2_smiles"])

train_df = train_df[(train_df["drug_1_smiles"]!="") & (train_df["drug_2_smiles"]!="")]
test_df  = test_df [(test_df ["drug_1_smiles"]!="") & (test_df ["drug_2_smiles"]!="")]

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

feats = {}

feats["morgan"]      = load_fp("../../data/features_morgan.csv")
feats["maccs"]       = load_fp("../../data/features_maccs.csv")
feats["rdkitfp"]     = load_fp("../../data/features_rdkitfp.csv")
feats["desc"]        = load_fp("../../data/features_descriptors.csv")
feats["mordred"]     = load_fp("../../data/features_mordred.csv")

print("✔ feature maps ready")

# =======================
# unified pooling function
# =======================
def fuse_features(row):
    vectors = []
    
    for name, fp_map in feats.items():
        fp1 = fp_map.get(row.drug_1_smiles)
        fp2 = fp_map.get(row.drug_2_smiles)
        if fp1 is None or fp2 is None:
            return None  # fail fast
        pooled = (fp1 + fp2) / 2
        vectors.append(pooled)
        
    return np.concatenate(vectors, axis=0)

# =======================
# convert DF into matrix
# =======================
def df_to_matrix(df):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        fused = fuse_features(row)
        if fused is None:
            continue
        X_list.append(fused)
        y_list.append(row.mean_reporting_frequency)
    return np.array(X_list), np.array(y_list)

X_train, y_train = df_to_matrix(train_df)
X_test,  y_test  = df_to_matrix(test_df)

print("Final feature vector size =", X_train.shape[1])
print("Train size =", X_train.shape)
print("Test size  =", X_test.shape)

# =======================
# train CatBoost
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

model.save_model(f"model_fold{fold_id}.cbm")
print(f"✔ Saved model_fold{fold_id}.cbm")

