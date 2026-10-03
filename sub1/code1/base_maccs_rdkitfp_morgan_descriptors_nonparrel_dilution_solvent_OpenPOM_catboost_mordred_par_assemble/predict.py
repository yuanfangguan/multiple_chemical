import pandas as pd
import os
import joblib
import numpy as np

# === Paths ===
TEST_PATH      = "test.csv"
MAP_PATH       = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH       = "../../data/raw/CID.csv"
RATA_RAW_PATH  = "../../data/raw/OpenPOM_Dream_RATA.csv"
MODEL_DIR      = "models/all_features_ensemble"
OUTPUT_CSV     = "predictions.csv"

# === Feature sets to load and merge ===
FEATURES_PATHS = {
    "mordred":    "../../data/raw/Mordred_Descriptors.csv",
    "maccs":      "../../data/processed/features_maccs.csv",
    "rdkitfp":    "../../data/processed/features_rdkitfp.csv",
    "morgan":     "../../data/processed/features_morgan.csv",
    "descriptors":"../../data/processed/features_descriptors.csv",
    "rata":       RATA_RAW_PATH
}

# === Solvent encoding ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}

def one_hot_encode_solvent(solvent):
    vec = np.zeros(len(PREDEFINED_SOLVENTS))
    idx = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[idx] = 1
    return vec

# === Load test & mapping data ===
df_test = pd.read_csv(TEST_PATH)
df_map  = pd.read_csv(MAP_PATH).fillna({"solvent": "unknown"})
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)
df_cid  = pd.read_csv(CID_PATH)
df_rata = pd.read_csv(RATA_RAW_PATH)

# Merge test with stimulus→molecule map
df_merged = df_test.merge(
    df_map[['stimulus', 'molecule', 'dilution', 'solvent_onehot']],
    on='stimulus', how='left'
)
stimulus_ids = df_merged['stimulus']
label_columns = [c for c in df_test.columns if c != 'stimulus']

# === Preprocess RATA ===
df_cid  = df_cid[df_cid.SMILES.notna()][['molecule', 'SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Merge all feature sets ===
df_all_feats = None
for name, path in FEATURES_PATHS.items():
    print(f"📦 Loading feature: {name}")
    if name == "rata":
        df_feat = df_rata_mapped.copy()
    elif "ordred" in name:
        # match your train.py logic
        df_feat = pd.read_csv(path, encoding="ISO-8859-1")
    else:
        df_feat = pd.read_csv(path)

    if "molecule" not in df_feat.columns:
        if "SMILES" in df_feat.columns:
            df_feat = df_feat.merge(df_cid, on="SMILES", how="left")
        else:
            raise ValueError(f"{name} features missing 'molecule' and 'SMILES'.")

    feat_cols = [c for c in df_feat.columns if c not in ("molecule", "SMILES")]
    df_feat = df_feat[["molecule"] + feat_cols]
    df_feat = df_feat.rename(columns={c: f"{name}_{c}" for c in feat_cols})

    df_all_feats = df_feat if df_all_feats is None else df_all_feats.merge(df_feat, on="molecule", how="outer")

# === Combine test data + features ===
df_final = df_merged.merge(df_all_feats, on="molecule", how="left")

# Expand solvent one‐hot into separate columns
solvent_array = np.vstack(df_final['solvent_onehot'])
solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
df_final = df_final.drop(columns=['solvent_onehot']).reset_index(drop=True)
df_final = pd.concat([df_final, solvent_df], axis=1)

# Drop unused columns
drop_cols = ['stimulus', 'molecule']
if 'SMILES' in df_final.columns:
    drop_cols.append('SMILES')
df_final = df_final.drop(columns=drop_cols)

# === Load feature list used in training ===
with open(os.path.join(MODEL_DIR, "features_used.txt")) as f:
    feature_columns = [line.strip() for line in f]

X_test = df_final[feature_columns]

# === Predict with each ensemble model ===
final_preds = pd.DataFrame()
for label in label_columns:
    print(f"🔮 Predicting for: {label}")
    ens_path = os.path.join(MODEL_DIR, f"ensemble_{label}.pkl")
    model = joblib.load(ens_path)
    final_preds[label] = model.predict(X_test)

# === Save to CSV ===
output = pd.concat([stimulus_ids.reset_index(drop=True), final_preds], axis=1)
output.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to: {OUTPUT_CSV}")

