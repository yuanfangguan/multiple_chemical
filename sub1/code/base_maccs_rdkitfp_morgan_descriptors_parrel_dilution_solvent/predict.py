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

# === Fixed solvent list ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}
num_solvent_types = len(PREDEFINED_SOLVENTS)

def one_hot_encode_solvent(solvent):
    vec = np.zeros(num_solvent_types)
    index = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[index] = 1
    return vec

# Shared data
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)

# Fill missing solvent values and one-hot encode
df_map["solvent"] = df_map["solvent"].fillna("unknown")
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)

# Merge in dilution and solvent one-hot
df_merged = df_test.merge(df_map[['stimulus', 'molecule', 'dilution', 'solvent_onehot']], on='stimulus', how='left')
stimulus_ids = df_merged['stimulus']
label_columns = [col for col in df_test.columns if col != 'stimulus']

# To store predictions from each model set
all_preds = {}

for name, feat_path in FEATURES_PATHS.items():
    print(f"\n📦 Using feature set: {name}")
    df_feats = pd.read_csv(feat_path)

    # Merge features
    df_final = df_merged.merge(df_feats, on='molecule', how='left')

    # Drop unused columns
    drop_cols = ['stimulus', 'molecule']
    if 'SMILES' in df_final.columns:
        drop_cols.append('SMILES')
    df_final = df_final.drop(columns=drop_cols)

    # Expand one-hot vectors into separate columns
    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    # Ensure dilution is numeric
    df_final['dilution'] = df_final['dilution'].astype(float)

    # Define feature matrix
    feature_columns = [col for col in df_final.columns if col not in label_columns]
    X_test = df_final[feature_columns]

    preds_df = pd.DataFrame()

    for label in label_columns:
        model_path = os.path.join(BASE_MODEL_DIR, name, f"model_{label}.pkl")
        print(f"🔍 Loading model: {model_path}")
        model = joblib.load(model_path)
        preds_df[label] = model.predict(X_test)

    all_preds[name] = preds_df

# === Average predictions ===
avg_preds = sum(all_preds.values()) / len(all_preds)

# Add stimulus column
final_predictions = pd.concat([stimulus_ids.reset_index(drop=True), avg_preds], axis=1)

# === Save ===
final_predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Averaged predictions saved to {OUTPUT_CSV}")

