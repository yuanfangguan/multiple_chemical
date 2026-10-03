import pandas as pd
import lightgbm as lgb
import os
import joblib  # for saving models

# Paths to your files
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_PATH = "../../../data_external/cid_smiles_hidden_features_layer11.csv"
OUTPUT_MODEL_DIR = "models"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Load datasets
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)
df_feats = pd.read_csv(FEATURES_PATH)

# Merge: train + map
df_merged = df_train.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')

# Merge: with features
df_final = df_merged.merge(df_feats, on='molecule', how='left')

# Drop unused columns
df_final = df_final.drop(columns=['stimulus', 'molecule'])

# Separate features and labels
label_columns = [col for col in df_train.columns if col != 'stimulus']
feature_columns = [col for col in df_final.columns if col not in label_columns and col !='SMILES']

X = df_final[feature_columns]
y = df_final[label_columns]

# Train a model per label and save each one
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
    
    # Save the model
    model_path = os.path.join(OUTPUT_MODEL_DIR, f"model_{label}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

print("✅ All models trained and saved.")

