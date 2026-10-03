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
BASE_MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Feature sets (excluding layer2–12, handled separately)
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

df_merged = df_test.merge(
    df_map[['stimulus', 'molecule', 'dilution', 'solvent_onehot']],
    on='stimulus', how='left'
)

# Add SMILES column to df_merged
df_cid = df_cid[df_cid['SMILES'].notna()][['molecule', 'SMILES']].drop_duplicates()
df_merged = df_merged.merge(df_cid, on='molecule', how='left')

stimulus_ids = df_merged['stimulus']
label_columns = [col for col in df_test.columns if col != 'stimulus']

# === Preprocess RATA feature
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Layer2–12 prediction and averaging
print("\n🧪 Averaging predictions from layer2–12 models")
LAYER_RANGE = range(2, 13)
layer_preds_collection = []

for layer in LAYER_RANGE:
    name = f"layer{layer}"
    model_dir = os.path.join(BASE_MODEL_DIR, name)
    feat_path = f"../../../data_external/50000_cid_smiles_hidden_features_{name}.csv"

    print(f"📦 Using feature set: {name}")
    df_feats = pd.read_csv(feat_path)

    if 'molecule' not in df_feats.columns and 'SMILES' in df_feats.columns:
        df_feats = df_feats.merge(df_cid, on='SMILES', how='left')

    if 'molecule' in df_feats.columns:
        df_final = df_merged.merge(df_feats, on='molecule', how='left')
    elif 'SMILES' in df_feats.columns:
        df_final = df_merged.merge(df_feats, on='SMILES', how='left')
    else:
        raise ValueError(f"{name} features missing both 'molecule' and 'SMILES'.")

    drop_cols = ['stimulus', 'molecule', 'SMILES']
    df_final = df_final.drop(columns=[col for col in drop_cols if col in df_final.columns])

    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    with open(os.path.join(model_dir, "features_used.txt")) as f:
        feature_columns = [line.strip() for line in f.readlines()]
    X_test = df_final[feature_columns]

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    layer_preds = pd.DataFrame()
    for model_file in model_files:
        label = model_file.replace("model_", "").replace(".pkl", "")
        model = joblib.load(os.path.join(model_dir, model_file))
        layer_preds[label] = model.predict(X_test)
    layer_preds_collection.append(layer_preds)

avg_layer_preds = sum(layer_preds_collection) / len(layer_preds_collection)
all_preds = {"layer2-12_avg": avg_layer_preds}

# === Other feature sets
for name, feat_path in FEATURES_PATHS.items():
    print(f"\n📦 Using feature set: {name}")
    model_dir = os.path.join(BASE_MODEL_DIR, name)

    if name == "rata":
        df_feats = df_rata_mapped.copy()
    else:
        df_feats = pd.read_csv(feat_path)

    if 'molecule' not in df_feats.columns and 'SMILES' in df_feats.columns:
        df_feats = df_feats.merge(df_cid, on='SMILES', how='left')

    if 'molecule' in df_feats.columns:
        df_final = df_merged.merge(df_feats, on='molecule', how='left')
    elif 'SMILES' in df_feats.columns:
        df_final = df_merged.merge(df_feats, on='SMILES', how='left')
    else:
        raise ValueError(f"{name} features missing both 'molecule' and 'SMILES'.")

    drop_cols = ['stimulus', 'molecule', 'SMILES']
    df_final = df_final.drop(columns=[col for col in drop_cols if col in df_final.columns])

    solvent_array = np.vstack(df_final['solvent_onehot'].values)
    solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
    df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
    df_final = pd.concat([df_final, solvent_df], axis=1)

    with open(os.path.join(model_dir, "features_used.txt")) as f:
        feature_columns = [line.strip() for line in f.readlines()]
    X_test = df_final[feature_columns]

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    current_preds = pd.DataFrame()
    for model_file in model_files:
        label = model_file.replace("model_", "").replace(".pkl", "")
        model = joblib.load(os.path.join(model_dir, model_file))
        current_preds[label] = model.predict(X_test)

    all_preds[name] = current_preds

# === Final ensemble
avg_preds = sum(all_preds.values()) / len(all_preds)

# === Output
final_predictions = pd.concat([stimulus_ids.reset_index(drop=True), avg_preds], axis=1)
final_predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Ensemble predictions saved to: {OUTPUT_CSV}")

