import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

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
    #"descriptors": "../../data/processed/features_descriptors.csv",
    "descriptors_more": "../../data/processed/features_descriptors_more.csv",
    "layered": "../../data/processed/features_layered.csv",
    "morgan_count": "../../data/processed/features_morgan_count.csv",
    "rata": RATA_RAW_PATH  # raw file, will handle specially
}

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

# === Preprocess RATA file once (SMILES → molecule) ===
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

    # Ensure molecule column present
    if 'molecule' not in df_feats.columns:
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
        else:
            raise ValueError(f"{name} feature set missing 'molecule' column and no SMILES to map")

    # Merge features with training
    df_final = df_merged.merge(df_feats, on='molecule', how='left')

    # Drop unnecessary columns
    drop_cols = ['stimulus', 'molecule']
    if 'SMILES' in df_final.columns:
        drop_cols.append('SMILES')
    df_final = df_final.drop(columns=drop_cols)

    # Expand solvent one-hot
    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    # Feature matrix
    feature_columns = [col for col in df_final.columns if col not in label_columns]
    X = df_final[feature_columns]
    y = df_final[label_columns]

    # Save feature list for use in prediction
    with open(os.path.join(model_dir, "features_used.txt"), "w") as f:
        for col in feature_columns:
            f.write(col + "\n")

    for label in label_columns:
        print(f"📈 Training model for: {label}")
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
        )
        model.fit(X, y[label])
        model_path = os.path.join(model_dir, f"model_{label}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model: {model_path}")

