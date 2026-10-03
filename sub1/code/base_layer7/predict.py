import pandas as pd
import lightgbm as lgb
import os
import joblib

# === File paths ===
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_PATH = "../../../data_external/cid_smiles_hidden_features_layer7.csv"
MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Load test data ===
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)
df_feats = pd.read_csv(FEATURES_PATH)

# Merge: test + map
df_merged = df_test.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')

# Merge: with features
df_final = df_merged.merge(df_feats, on='molecule', how='left')

# Save stimulus column for output
stimulus_ids = df_final['stimulus']

# Drop unused columns
df_final = df_final.drop(columns=['stimulus', 'molecule'])

# Prepare feature matrix
label_columns = [col for col in df_test.columns if col != 'stimulus']
feature_columns = [col for col in df_final.columns if col not in label_columns and col !='SMILES']
X_test = df_final[feature_columns]

# === Load models and make predictions ===
predictions = pd.DataFrame({'stimulus': stimulus_ids})

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        label = model_file.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"Loading model: {model_path}")
        
        model = joblib.load(model_path)
        preds = model.predict(X_test)
        predictions[label] = preds

# === Save predictions ===
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")

