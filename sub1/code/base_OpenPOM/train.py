import pandas as pd
import lightgbm as lgb
import os
import joblib

# === Paths ===
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
MODEL_DIR = "models_rata_debug_only"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load data ===
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)
df_cid = pd.read_csv(CID_PATH)
df_rata = pd.read_csv(RATA_RAW_PATH)

# === Map stimulus → molecule, then molecule → SMILES
df_meta = df_train.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')
df_cid = df_cid[df_cid['SMILES'].notna()][['molecule', 'SMILES']].drop_duplicates()

# === Map RATA via SMILES → molecule
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Merge in RATA features with training data
df_full = df_meta.merge(df_rata_mapped, on='molecule', how='left')

# === Drop unnecessary columns
df_full = df_full.drop(columns=['stimulus', 'molecule', 'SMILES'])

# === Define features and labels
label_columns = [col for col in df_train.columns if col != 'stimulus']
feature_columns = [col for col in df_full.columns if col not in label_columns]

# Save feature columns for prediction-time use
with open(os.path.join(MODEL_DIR, "features_used.txt"), "w") as f:
    for col in feature_columns:
        f.write(col + "\n")

X = df_full[feature_columns]
y = df_full[label_columns]

# === Train one model per label
for label in label_columns:
    print(f"📈 Training model for: {label}")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        random_state=42
    )
    model.fit(X, y[label])

    model_path = os.path.join(MODEL_DIR, f"model_{label}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved model: {model_path}")

