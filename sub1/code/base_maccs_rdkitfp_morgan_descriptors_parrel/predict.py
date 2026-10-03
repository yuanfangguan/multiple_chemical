import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# Paths
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv"
}
BASE_MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# Shared data
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)
df_merged = df_test.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')
stimulus_ids = df_merged['stimulus']
label_columns = [col for col in df_test.columns if col != 'stimulus']

# To store predictions from each model set
all_preds = {}

for name, feat_path in FEATURES_PATHS.items():
    print(f"\n📦 Using feature set: {name}")
    df_feats = pd.read_csv(feat_path)
    df_final = df_merged.merge(df_feats, on='molecule', how='left')
    df_final = df_final.drop(columns=['stimulus', 'molecule'])

    feature_columns = [col for col in df_final.columns if col not in label_columns and col != 'SMILES']
    X_test = df_final[feature_columns]

    preds_df = pd.DataFrame()

    for label in label_columns:
        model_path = os.path.join(BASE_MODEL_DIR, name, f"model_{label}.pkl")
        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)
        preds_df[label] = model.predict(X_test)

    all_preds[name] = preds_df

# === Average predictions ===
avg_preds = sum(all_preds.values()) / len(all_preds)

# Add stimulus column
final_predictions = pd.concat([stimulus_ids, avg_preds], axis=1)

# === Save ===
final_predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Averaged predictions saved to {OUTPUT_CSV}")

