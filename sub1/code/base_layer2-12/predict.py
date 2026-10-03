import pandas as pd
import lightgbm as lgb
import os
import joblib
from functools import reduce

# === File paths ===
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_BASE_PATH = "../../../data_external"
MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Load test data ===
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)

# Merge test + map to get molecule
df_merged = df_test.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')

# === Load and merge features from layer2 to layer12 ===
feature_dfs = []
for layer in range(2, 13):
    path = os.path.join(FEATURES_BASE_PATH, f"cid_smiles_hidden_features_layer{layer}.csv")
    df_layer = pd.read_csv(path).drop(columns=['SMILES'], errors='ignore')  # Drop SMILES if present

    # Rename feature columns to indicate layer
    df_layer = df_layer.rename(columns={
        col: f"{col}_layer{layer}" for col in df_layer.columns if col != 'molecule'
    })

    feature_dfs.append(df_layer)

# Merge all layer feature DataFrames
df_feats_all = reduce(lambda left, right: pd.merge(left, right, on='molecule', how='outer'), feature_dfs)

# Merge with test data
df_final = df_merged.merge(df_feats_all, on='molecule', how='left')

# Save stimulus column for output
stimulus_ids = df_final['stimulus']

# Drop unused columns
df_final = df_final.drop(columns=['stimulus', 'molecule'])

# Determine features (exclude any columns not from features)
label_columns = [col for col in df_test.columns if col != 'stimulus']
feature_columns = [col for col in df_final.columns if col not in label_columns]

X_test = df_final[feature_columns]

# === Load models and make predictions ===
predictions = pd.DataFrame({'stimulus': stimulus_ids})

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        label = model_file.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"🔍 Loading model: {model_path}")

        model = joblib.load(model_path)
        preds = model.predict(X_test)
        predictions[label] = preds

# === Save predictions ===
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")

