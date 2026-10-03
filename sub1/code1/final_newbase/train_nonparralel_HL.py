import pandas as pd
from functools import reduce
from catboost import CatBoostRegressor
import os
import joblib
import numpy as np
os.system("rm -rf model*")

# === Paths ===
TRAIN_PATH       = "train.csv"
MAP_PATH         = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH         = "../../data/raw/CID.csv"
RATA_RAW_PATH    = "../../data/raw/OpenPOM_Dream_RATA.csv"
BASE_MODEL_DIR   = "models_combined"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# === Feature paths ===
FEATURES_PATHS = {
    "maccs":       "../../data/processed/features_maccs.csv",
    "rdkitfp":     "../../data/processed/features_rdkitfp.csv",
    "morgan":      "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "rata":        RATA_RAW_PATH
}

# === Solvent one-hot encoding ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}
def one_hot_encode_solvent(solvent):
    vec = np.zeros(len(PREDEFINED_SOLVENTS))
    idx = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[idx] = 1
    return vec

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_map   = pd.read_csv(MAP_PATH)
df_cid   = pd.read_csv(CID_PATH)

df_map["solvent"] = df_map["solvent"].fillna("unknown")
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)
df_map["intensity_binary"] = df_map["Intensity_label"].map({"H": 1, "L": 0})

# Merge train with map info
label_columns = [c for c in df_train.columns if c != 'stimulus']
df_merged = (
    df_train
    .merge(df_map[['stimulus','molecule','dilution','solvent_onehot']], on='stimulus', how='left')
    .merge(df_map[['stimulus','intensity_binary']], on='stimulus', how='left')
)

# === Preprocess RATA ===
# Map SMILES to molecule
df_rata = pd.read_csv(RATA_RAW_PATH)
df_cid  = df_cid[df_cid['SMILES'].notna()][['molecule','SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Load & prefix each feature set ===
feature_dfs = []
for name, path in FEATURES_PATHS.items():
    if name == 'rata':
        df_feats = df_rata_mapped.copy()
    else:
        df_feats = pd.read_csv(path)

    # Ensure 'molecule' column exists
    if 'molecule' not in df_feats.columns:
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
        else:
            raise ValueError(f"Feature set '{name}' missing 'molecule' and no SMILES to map.")

    # Drop SMILES after mapping
    df_feats = df_feats.drop(columns=[c for c in ['SMILES'] if c in df_feats.columns])

    # Prefix feature columns (except 'molecule') to avoid collisions
    feat_cols = [c for c in df_feats.columns if c != 'molecule']
    df_feats = (
        df_feats
        .set_index('molecule')[feat_cols]
        .add_prefix(f"{name}_")
        .reset_index()
    )
    feature_dfs.append(df_feats)

# Merge all feature sets on 'molecule'
df_all_feats = reduce(
    lambda left, right: left.merge(right, on='molecule', how='outer'),
    feature_dfs
)

# === Final merge ===
# Merge combined features with training metadata
# Use inner join to keep only molecules present in both

df_final = df_merged.merge(df_all_feats, on='molecule', how='inner')

# Drop ID columns
drop_cols = ['stimulus','molecule']
if 'SMILES' in df_final.columns:
    drop_cols.append('SMILES')
df_final = df_final.drop(columns=drop_cols)

# Expand solvent one-hot
solvent_array = np.vstack(df_final['solvent_onehot'].values)
solv_df = pd.DataFrame(solvent_array, columns=[f'solvent_{s}' for s in PREDEFINED_SOLVENTS])
df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
df_final = pd.concat([df_final, solv_df], axis=1)

# Prepare X, y
feature_columns = [c for c in df_final.columns if c not in label_columns]
X = df_final[feature_columns]
y = df_final[label_columns]

# Save feature list
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
with open(os.path.join(BASE_MODEL_DIR, 'features_used.txt'), 'w') as f:
    for col in feature_columns:
        f.write(col + '\n')

# === Train models on combined features ===
for label in label_columns:
    print(f"📈 Training combined model for: {label}")
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        random_seed=42,
        verbose=100
    )
    model.fit(X, y[label])
    model_path = os.path.join(BASE_MODEL_DIR, f"model_combined_{label}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved combined model: {model_path}")

