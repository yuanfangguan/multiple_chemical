import pandas as pd
import os
import joblib
import numpy as np

# === Paths ===
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
BASE_MODEL_DIR = "models"
OUTPUT_CSV = "predictions_HLnomodred.csv"

# === Feature sets ===
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
# Binarize intensity label: H=1, L=0
df_map["intensity_binary"] = df_map["Intensity_label"].map({"H": 1, "L": 0})
df_merged = df_merged.merge(df_map[['stimulus', 'intensity_binary']], on='stimulus', how='left')

stimulus_ids = df_merged['stimulus']
label_columns = [col for col in df_test.columns if col != 'stimulus']

# === Preprocess RATA feature
df_cid = df_cid[df_cid['SMILES'].notna()][['molecule', 'SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Collect predictions from each feature set
all_preds = {}

for name, feat_path in FEATURES_PATHS.items():
    print(f"\n📦 Using feature set: {name}")
    model_dir = os.path.join(BASE_MODEL_DIR, name)

    if name == "rata":
        df_feats = df_rata_mapped.copy()
    else:
        df_feats = pd.read_csv(feat_path)

    if 'molecule' not in df_feats.columns:
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.merge(df_cid, on='SMILES', how='left')
        else:
            raise ValueError(f"{name} features missing 'molecule' and 'SMILES'.")

    df_final = df_merged.merge(df_feats, on='molecule', how='left')

    # Drop unnecessary columns
    drop_cols = ['stimulus', 'molecule']
    if 'SMILES' in df_final.columns:
        drop_cols.append('SMILES')
    df_final = df_final.drop(columns=drop_cols)

    # Expand solvent one-hot
    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    # Load feature list from training
    with open(os.path.join(model_dir, "features_used.txt")) as f:
        feature_columns = [line.strip() for line in f.readlines()]
    X_test = df_final[feature_columns]

    # Load model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    current_preds = pd.DataFrame()
    for model_file in model_files:
        label = model_file.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)
        current_preds[label] = model.predict(X_test)

    all_preds[name] = current_preds

# === Ensemble: average all model outputs
avg_preds = sum(all_preds.values()) / len(all_preds)

# === Output results
final_predictions = pd.concat([stimulus_ids.reset_index(drop=True), avg_preds], axis=1)
final_predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Ensemble predictions saved to: {OUTPUT_CSV}")

