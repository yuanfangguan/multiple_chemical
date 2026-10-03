import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# === Paths ===
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
FEATURE_MODEL_DIR = "models_combined"  # ← match combined model dir
OUTPUT_CSV = "predictions_combined.csv"

# === Feature paths ===
FEATURES_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "rata": RATA_RAW_PATH
}

# === Solvent encoding ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}
def one_hot_encode_solvent(solvent):
    vec = np.zeros(len(PREDEFINED_SOLVENTS))
    index = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[index] = 1
    return vec

# === Load shared test data ===
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)
df_cid = pd.read_csv(CID_PATH)
df_rata = pd.read_csv(RATA_RAW_PATH)

df_map["solvent"] = df_map["solvent"].fillna("unknown")
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)
df_merged = df_test.merge(df_map[['stimulus', 'molecule', 'dilution', 'solvent_onehot']], on='stimulus', how='left')
stimulus_ids = df_merged['stimulus']

# === Preprocess RATA
df_cid = df_cid[df_cid['SMILES'].notna()][['molecule', 'SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Merge all features by molecule
dfs_features = []
for name, feat_path in FEATURES_PATHS.items():
    print(f"🔄 Loading feature set: {name}")
    if name == "rata":
        df_feat = df_rata_mapped.copy()
    else:
        df_feat = pd.read_csv(feat_path)

    if 'molecule' not in df_feat.columns:
        if 'SMILES' in df_feat.columns:
            df_feat = df_feat.merge(df_cid, on='SMILES', how='left')
        else:
            raise ValueError(f"{name} features missing 'molecule' and 'SMILES'.")

    feature_cols = [col for col in df_feat.columns if col not in ['molecule', 'SMILES']]
    df_feat = df_feat[['molecule'] + feature_cols]
    df_feat = df_feat.rename(columns={col: f"{name}_{col}" for col in feature_cols})
    dfs_features.append(df_feat)

# Combine all features on 'molecule'
from functools import reduce
df_all_features = reduce(lambda left, right: pd.merge(left, right, on='molecule', how='outer'), dfs_features)

# Merge with test data
df_final = df_merged.merge(df_all_features, on='molecule', how='left')

# Expand solvent one-hot
solvent_array = np.vstack(df_final['solvent_onehot'].values)
solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
df_final = pd.concat([df_final, solvent_df], axis=1)

# Drop unnecessary columns
drop_cols = ['stimulus', 'molecule']
if 'SMILES' in df_final.columns:
    drop_cols.append('SMILES')
df_final = df_final.drop(columns=drop_cols)

# Load feature list used in training
with open(os.path.join(FEATURE_MODEL_DIR, "features_used.txt")) as f:
    feature_columns = [line.strip() for line in f.readlines()]
X_test = df_final[feature_columns]

# Load models and predict
preds = pd.DataFrame()
model_files = [f for f in os.listdir(FEATURE_MODEL_DIR) if f.endswith(".pkl")]
for model_file in model_files:
    label = model_file.replace("model_", "").replace(".pkl", "")
    model = joblib.load(os.path.join(FEATURE_MODEL_DIR, model_file))
    preds[label] = model.predict(X_test)

# Save predictions
final_df = pd.concat([stimulus_ids.reset_index(drop=True), preds], axis=1)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to: {OUTPUT_CSV}")

