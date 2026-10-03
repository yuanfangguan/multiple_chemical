### Updated Full Training Code with Concatenated Feature Chunks ###

import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np
from functools import reduce

# === Paths ===
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
BASE_MODEL_DIR = "models"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# === Feature paths ===
FEATURES_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "rata": RATA_RAW_PATH
}

for i in range(2, 13):
    FEATURES_PATHS[f"layer{i}"] = f"../../../data_external/50000_cid_smiles_hidden_features_layer{i}.csv"
    FEATURES_PATHS[f"combined_layer{i}"] = f"/local/disk3/gyuanfan/Olfactory_2025/data_rgerkin/sub1_cid_smiles_hidden_features_layer{i}.csv"

# === Solvent one-hot encoding ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}

def one_hot_encode_solvent(solvent):
    vec = np.zeros(len(PREDEFINED_SOLVENTS))
    index = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[index] = 1
    return vec

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)
df_cid = pd.read_csv(CID_PATH)
df_map["solvent"] = df_map["solvent"].fillna("unknown")
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)
df_merged = df_train.merge(df_map[['stimulus', 'molecule', 'dilution', 'solvent_onehot']], on='stimulus', how='left')
label_columns = [col for col in df_train.columns if col != 'stimulus']

# === Preprocess RATA file once ===
df_rata = pd.read_csv(RATA_RAW_PATH)
df_cid = df_cid[df_cid['SMILES'].notna()][['molecule', 'SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Train models for each feature set ===
for name, feat_path in FEATURES_PATHS.items():
    print(f"\n🔧 Training models on: {name}")
    model_dir = os.path.join(BASE_MODEL_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    if name == "rata":
        df_feats = df_rata_mapped.copy()
    else:
        df_feats = pd.read_csv(feat_path)

    if 'molecule' not in df_feats.columns:
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
        else:
            raise ValueError(f"{name} feature set missing 'molecule' column and no SMILES to map")

    df_final = df_merged.merge(df_feats, on='molecule', how='left')
    drop_cols = ['stimulus', 'molecule']
    if 'SMILES' in df_final.columns:
        drop_cols.append('SMILES')
    df_final = df_final.drop(columns=drop_cols)

    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    feature_columns = [col for col in df_final.columns if col not in label_columns]
    X = df_final[feature_columns]
    y = df_final[label_columns]

    with open(os.path.join(model_dir, "features_used.txt"), "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    for label in label_columns:
        print(f"📈 Training model for: {label}")
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, random_state=42)
        model.fit(X, y[label])
        model_path = os.path.join(model_dir, f"model_{label}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model: {model_path}")

# === Train on concatenated, randomized, chunked features ===
print("\n🔧 Training models on concatenated features (chunked by 1000)")

feature_dfs = []
for name, feat_path in FEATURES_PATHS.items():
    if name == "rata":
        df_feats = df_rata_mapped.copy()
    else:
        df_feats = pd.read_csv(feat_path)
    if 'molecule' not in df_feats.columns:
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
        else:
            raise ValueError(f"{name} feature set missing 'molecule' column and no SMILES to map")
    feature_dfs.append(df_feats)

all_layers = list(range(2, 13))
for i in all_layers:
    df_feats = pd.read_csv(f"../../../data_external/50000_cid_smiles_hidden_features_layer{i}.csv")
    if 'molecule' not in df_feats.columns and 'SMILES' in df_feats.columns:
        df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
    feature_dfs.append(df_feats)

for i in all_layers:
    df_feats = pd.read_csv(f"/local/disk3/gyuanfan/Olfactory_2025/data_rgerkin/sub1_cid_smiles_hidden_features_layer{i}.csv")
    if 'molecule' not in df_feats.columns and 'SMILES' in df_feats.columns:
        df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
    feature_dfs.append(df_feats)

from functools import reduce
df_all_feats = reduce(lambda left, right: pd.merge(left, right, on='molecule', how='outer'), feature_dfs)
df_final = df_merged.merge(df_all_feats, on='molecule', how='left')

if 'SMILES' in df_final.columns:
    df_final = df_final.drop(columns=['stimulus', 'molecule', 'SMILES'])
else:
    df_final = df_final.drop(columns=['stimulus', 'molecule'])

solvent_array = np.vstack(df_final['solvent_onehot'].values)
solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
df_final = pd.concat([df_final, solvent_df], axis=1)

feature_columns = [col for col in df_final.columns if col not in label_columns]
np.random.seed(42)
shuffled_features = np.random.permutation(feature_columns)

# Save the order for test phase
with open(os.path.join(BASE_MODEL_DIR, "feature_order.txt"), "w") as f:
    for col in shuffled_features:
        f.write(col + "\n")

X = df_final[shuffled_features]
y = df_final[label_columns]

chunk_size = 1000
num_chunks = (X.shape[1] + chunk_size - 1) // chunk_size

for chunk_idx in range(num_chunks):
    chunk_features = shuffled_features[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
    chunk_dir = os.path.join(BASE_MODEL_DIR, f"concat_chunk_{chunk_idx}")
    os.makedirs(chunk_dir, exist_ok=True)

    with open(os.path.join(chunk_dir, "features_used.txt"), "w") as f:
        for col in chunk_features:
            f.write(col + "\n")

    for label in label_columns:
        print(f"📈 Training chunk {chunk_idx}, label {label}")
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, random_state=42)
        model.fit(X[list(chunk_features)], y[label])
        model_path = os.path.join(chunk_dir, f"model_{label}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model: {model_path}")

