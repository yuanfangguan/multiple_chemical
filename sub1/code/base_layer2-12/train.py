import pandas as pd
import lightgbm as lgb
import os
import joblib
from functools import reduce

# Paths
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_BASE_PATH = "../../../data_external"
OUTPUT_MODEL_DIR = "models"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Load base data
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)

# Merge train and map
df_merged = df_train.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')

# Load and concatenate features from layer2 to layer12
feature_dfs = []
for layer in range(2, 13):
    path = os.path.join(FEATURES_BASE_PATH, f"cid_smiles_hidden_features_layer{layer}.csv")
    df_layer = pd.read_csv(path).drop(columns=['SMILES'], errors='ignore')  # Drop SMILES if exists

    # Rename columns to include layer info
    df_layer = df_layer.rename(columns={
        col: f"{col}_layer{layer}" for col in df_layer.columns if col != 'molecule'
    })

    feature_dfs.append(df_layer)

# Merge all feature layers on 'molecule'
df_feats_all = reduce(lambda left, right: pd.merge(left, right, on='molecule', how='outer'), feature_dfs)

# Merge features into training data
df_final = df_merged.merge(df_feats_all, on='molecule', how='left')

# Drop unused columns
df_final = df_final.drop(columns=['stimulus', 'molecule'])

# Separate features and labels
label_columns = [col for col in df_train.columns if col != 'stimulus']
feature_columns = [col for col in df_final.columns if col not in label_columns]

X = df_final[feature_columns]
y = df_final[label_columns]

# Train a model per label
for label in label_columns:
    print(f"Training model for: {label}")
    y_label = y[label]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        random_state=42
    )

    model.fit(X, y_label)

    model_path = os.path.join(OUTPUT_MODEL_DIR, f"model_{label}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

print("✅ All models trained and saved with layer 2–12 features.")

