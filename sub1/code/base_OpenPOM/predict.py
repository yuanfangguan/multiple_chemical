import pandas as pd
import lightgbm as lgb
import os
import joblib

# === Paths ===
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
MODEL_DIR = "models_rata_debug_only"
OUTPUT_CSV = "predictions.csv"

# === Load test, map, cid, and rata
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)
df_cid = pd.read_csv(CID_PATH)
df_rata = pd.read_csv(RATA_RAW_PATH)

# === Map stimulus → molecule → SMILES
df_meta = df_test.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')
df_cid = df_cid[df_cid['SMILES'].notna()][['molecule', 'SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Merge with RATA features
df_full = df_meta.merge(df_rata_mapped, on='molecule', how='left')

# === Drop unused columns
stimulus_ids = df_full['stimulus']
df_full = df_full.drop(columns=['stimulus', 'molecule', 'SMILES'])

# === Load feature columns from training
with open(os.path.join(MODEL_DIR, "features_used.txt")) as f:
    feature_columns = [line.strip() for line in f.readlines()]

# Keep only columns that were used in training
X_test = df_full[feature_columns]

# === Infer labels from saved models
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_") and f.endswith(".pkl")]
label_columns = [f.replace("model_", "").replace(".pkl", "") for f in model_files]

# === Predict
preds_df = pd.DataFrame()
for label in label_columns:
    model_path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
    print(f"🔍 Loading model: {model_path}")
    model = joblib.load(model_path)
    preds_df[label] = model.predict(X_test)

# === Save predictions
final_predictions = pd.concat([stimulus_ids.reset_index(drop=True), preds_df], axis=1)
final_predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ RATA-only predictions saved to {OUTPUT_CSV}")

