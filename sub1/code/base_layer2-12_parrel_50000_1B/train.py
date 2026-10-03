import pandas as pd
import lightgbm as lgb
import os
import joblib

# Paths
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_BASE_PATH = "../../../data_external"
OUTPUT_DIR = "models_per_layer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load training data
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)
df_merged = df_train.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')

# Define label columns
label_columns = [col for col in df_train.columns if col != 'stimulus']

# Train per layer
for layer in range(2, 13):
    print(f"📚 Training models for Layer {layer}")
    layer_model_dir = os.path.join(OUTPUT_DIR, f"layer{layer}")
    os.makedirs(layer_model_dir, exist_ok=True)

    feature_path = os.path.join(FEATURES_BASE_PATH, f"1B-50000_cid_smiles_hidden_features_layer{layer}.csv")
    df_feats = pd.read_csv(feature_path).drop(columns=['SMILES'], errors='ignore')
    df_layer = df_merged.merge(df_feats, on='molecule', how='left')

    df_layer = df_layer.drop(columns=['stimulus', 'molecule'])
    feature_columns = [col for col in df_layer.columns if col not in label_columns]

    X = df_layer[feature_columns]
    y = df_layer[label_columns]

    for label in label_columns:
        print(f" - Training model for label: {label}")
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
        )
        model.fit(X, y[label])
        model_path = os.path.join(layer_model_dir, f"model_{label}.pkl")
        joblib.dump(model, model_path)
        print(f"   ✔️ Saved to {model_path}")

print("✅ All models for all layers trained and saved.")

