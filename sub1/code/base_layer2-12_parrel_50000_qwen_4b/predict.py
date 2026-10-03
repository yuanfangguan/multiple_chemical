import pandas as pd
import lightgbm as lgb
import os
import joblib

# Paths
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_BASE_PATH = "../../../data_external"
MODEL_BASE_DIR = "models_per_layer"
OUTPUT_CSV = "predictions.csv"

# Load test + map
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)
df_merged = df_test.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')

# Store predictions for each layer
layer_preds_list = []

for layer in range(2, 13):
    print(f"🔍 Predicting with Layer {layer}")
    feature_path = os.path.join(FEATURES_BASE_PATH, f"Qwen3-4b-50000_cid_smiles_hidden_features_layer{layer}.csv")
    df_feats = pd.read_csv(feature_path).drop(columns=['SMILES'], errors='ignore')
    df_layer = df_merged.merge(df_feats, on='molecule', how='left')

    stimulus_ids = df_layer['stimulus']
    df_layer = df_layer.drop(columns=['stimulus', 'molecule'])
    feature_columns = [col for col in df_layer.columns if col not in df_test.columns]
    X_test = df_layer[feature_columns]

    model_dir = os.path.join(MODEL_BASE_DIR, f"layer{layer}")
    layer_preds = pd.DataFrame({'stimulus': stimulus_ids})

    for model_file in os.listdir(model_dir):
        if model_file.endswith(".pkl"):
            label = model_file.replace("model_", "").replace(".pkl", "")
            model_path = os.path.join(model_dir, model_file)
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            layer_preds[label] = preds

    layer_preds_list.append(layer_preds)

# Average predictions across layers
print("📊 Averaging predictions across layers...")
predictions = layer_preds_list[0].copy()
label_columns = [col for col in predictions.columns if col != 'stimulus']

for label in label_columns:
    preds_sum = sum([df[label] for df in layer_preds_list])
    predictions[label] = preds_sum / len(layer_preds_list)

# Save final predictions
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Averaged predictions saved to {OUTPUT_CSV}")

