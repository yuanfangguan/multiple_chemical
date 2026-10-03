import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# Paths
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv"
}
BASE_MODEL_DIR = "models"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# === Fixed solvent list ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}
num_solvent_types = len(PREDEFINED_SOLVENTS)

def one_hot_encode_solvent(solvent):
    vec = np.zeros(num_solvent_types)
    index = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[index] = 1
    return vec

# Load shared data
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)

# Fill missing solvent values and one-hot encode
df_map["solvent"] = df_map["solvent"].fillna("unknown")
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)

# Merge in dilution and solvent info
df_merged = df_train.merge(df_map[['stimulus', 'molecule', 'dilution', 'solvent_onehot']], on='stimulus', how='left')
label_columns = [col for col in df_train.columns if col != 'stimulus']

for name, feat_path in FEATURES_PATHS.items():
    print(f"\n🔧 Training models on: {name}")
    
    # Prepare output model dir
    model_dir = os.path.join(BASE_MODEL_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    df_feats = pd.read_csv(feat_path)
    df_final = df_merged.merge(df_feats, on='molecule', how='left')

    # Drop unnecessary columns
    drop_cols = ['stimulus', 'molecule']
    if 'SMILES' in df_final.columns:
        drop_cols.append('SMILES')
    df_final = df_final.drop(columns=drop_cols)

    # Expand one-hot vectors into separate columns
    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    # Use dilution and one-hot solvent + fingerprint as features
    feature_columns = [col for col in df_final.columns if col not in label_columns]
    X = df_final[feature_columns]
    y = df_final[label_columns]

    for label in label_columns:
        print(f"Training model for: {label}")
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

