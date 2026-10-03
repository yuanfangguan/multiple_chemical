import pandas as pd
import lightgbm as lgb
import os
import joblib

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

# Load shared data
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)
df_merged = df_train.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')
label_columns = [col for col in df_train.columns if col != 'stimulus']

for name, feat_path in FEATURES_PATHS.items():
    print(f"\n🔧 Training models on: {name}")
    
    # Prepare output model dir
    model_dir = os.path.join(BASE_MODEL_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    df_feats = pd.read_csv(feat_path)
    df_final = df_merged.merge(df_feats, on='molecule', how='left')
    df_final = df_final.drop(columns=['stimulus', 'molecule'])
    
    feature_columns = [col for col in df_final.columns if col not in label_columns and col != 'SMILES']
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

