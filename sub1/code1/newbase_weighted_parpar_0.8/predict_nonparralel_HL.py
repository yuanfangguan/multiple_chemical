import pandas as pd
from functools import reduce
import os
import joblib
import numpy as np

# === Paths ===
TEST_PATH     = "test.csv"
MAP_PATH      = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH      = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
MODEL_DIR     = "models_combined"
OUTPUT_CSV    = "predictions_nonparralel_HL.csv"

# === Feature sources ===
FEATURES_PATHS = {
    "maccs":       "../../data/processed/features_maccs.csv",
    "rdkitfp":     "../../data/processed/features_rdkitfp.csv",
    "morgan":      "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "rata":        RATA_RAW_PATH
}

# === Solvent one-hot ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}
def one_hot_encode_solvent(solvent):
    vec = np.zeros(len(PREDEFINED_SOLVENTS))
    idx = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[idx] = 1
    return vec

# === Load test and mapping ===
df_test = pd.read_csv(TEST_PATH)
df_map  = pd.read_csv(MAP_PATH)
df_cid  = pd.read_csv(CID_PATH)

# Fill missing solvents and encode
[df_map["solvent"]] = [df_map["solvent"].fillna("unknown")]
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)
# Binarize intensity label
df_map["intensity_binary"] = df_map["Intensity_label"].map({"H":1, "L":0})

# Merge test with map metadata
df_merged = (
    df_test
    .merge(df_map[['stimulus','molecule','dilution','solvent_onehot']], on='stimulus', how='left')
    .merge(df_map[['stimulus','intensity_binary']], on='stimulus', how='left')
)
stimulus_ids = df_merged['stimulus']
label_columns = [c for c in df_test.columns if c != 'stimulus']

# === Preprocess RATA ===
df_rata = pd.read_csv(RATA_RAW_PATH)
df_cid_map = df_cid[df_cid['SMILES'].notna()][['molecule','SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid_map, on='SMILES', how='inner')

# === Load and prefix each feature set ===
feature_dfs = []
for name, path in FEATURES_PATHS.items():
    if name == 'rata':
        df_feats = df_rata_mapped.copy()
    else:
        df_feats = pd.read_csv(path)

    # Ensure 'molecule'
    if 'molecule' not in df_feats.columns:
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.merge(df_cid_map, on='SMILES', how='left')
        else:
            raise ValueError(f"Feature set '{name}' missing 'molecule' and no SMILES to map.")

    # Drop SMILES after mapping
    df_feats = df_feats.drop(columns=[c for c in ['SMILES'] if c in df_feats.columns])

    # Prefix feature columns to avoid collisions
    feat_cols = [c for c in df_feats.columns if c != 'molecule']
    df_pref = (
        df_feats
        .set_index('molecule')[feat_cols]
        .add_prefix(f"{name}_")
        .reset_index()
    )
    feature_dfs.append(df_pref)

# Merge all prefixed feature sets
df_all_feats = reduce(lambda a, b: a.merge(b, on='molecule', how='outer'), feature_dfs)

# === Final merge with test metadata ===
df_final = df_merged.merge(df_all_feats, on='molecule', how='inner')

# Drop identifier columns
drop_cols = ['stimulus', 'molecule']
if 'SMILES' in df_final.columns:
    drop_cols.append('SMILES')
df_final = df_final.drop(columns=drop_cols)

# Expand solvent one-hot into separate columns
sol_arr = np.vstack(df_final['solvent_onehot'].values)
sol_df  = pd.DataFrame(sol_arr, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
df_final = pd.concat([df_final, sol_df], axis=1)

# Prepare test features X
with open(os.path.join(MODEL_DIR, 'features_used.txt')) as f:
    feature_columns = [line.strip() for line in f.readlines()]
X_test = df_final[feature_columns]

# === Load combined models and predict ===
pred_df = pd.DataFrame()
for model_file in os.listdir(MODEL_DIR):
    if model_file.startswith('model_combined_') and model_file.endswith('.pkl'):
        label = model_file.replace('model_combined_', '').replace('.pkl', '')
        model = joblib.load(os.path.join(MODEL_DIR, model_file))
        pred_df[label] = model.predict(X_test)

# Assemble and save predictions
output = pd.concat([stimulus_ids.reset_index(drop=True), pred_df], axis=1)
output.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Combined-ensemble predictions saved to: {OUTPUT_CSV}")

