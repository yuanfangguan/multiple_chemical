import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import os
import joblib
import numpy as np
os.system("rm -rf model*")

# === Paths ===
TRAIN_PATH = "train.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
CID_PATH = "../../data/raw/CID.csv"
RATA_RAW_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
BASE_MODEL_DIR = "models/all_features_ensemble"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# === Feature paths ===
FEATURES_PATHS = {
    "mordred": "../../data/raw/Mordred_Descriptors.csv",
    "maccs": "../../data/processed/features_maccs.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "rata": RATA_RAW_PATH
}

# === Solvent one-hot ===
PREDEFINED_SOLVENTS = ["DEP", "NT", "PG", "solvent", "unknown"]
solvent_to_onehot = {s: i for i, s in enumerate(PREDEFINED_SOLVENTS)}
def one_hot_encode_solvent(solvent):
    vec = np.zeros(len(PREDEFINED_SOLVENTS))
    idx = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[idx] = 1
    return vec

# === Load data ===
df_train = pd.read_csv(TRAIN_PATH)
df_map   = pd.read_csv(MAP_PATH).fillna({"solvent":"unknown"})
df_cid   = pd.read_csv(CID_PATH)
df_map["solvent_onehot"] = df_map["solvent"].apply(one_hot_encode_solvent)

df_merged = df_train.merge(
    df_map[['stimulus','molecule','dilution','solvent_onehot']],
    on='stimulus', how='left'
)
label_columns = [c for c in df_train if c != 'stimulus']

# === Preprocess RATA ===
df_rata = pd.read_csv(RATA_RAW_PATH)
df_cid  = df_cid[df_cid.SMILES.notna()][['molecule','SMILES']].drop_duplicates()
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')

# === Merge all feature sets ===
df_all_feats = None
for name, path in FEATURES_PATHS.items():
    print(f"📥 Loading: {name}")
    if name == "rata":
        df_feat = df_rata_mapped.copy()
    elif "ordred" in name:
        df_feat = pd.read_csv(path, encoding="ISO-8859-1")
    else:
        df_feat = pd.read_csv(path)
    # ensure molecule column
    if "molecule" not in df_feat:
        df_feat = df_feat.merge(df_cid, on="SMILES", how="left")
    feat_cols = [c for c in df_feat if c not in ("molecule","SMILES")]
    df_feat = df_feat[["molecule"] + feat_cols]
    df_feat = df_feat.rename(columns={c: f"{name}_{c}" for c in feat_cols})
    df_all_feats = df_feat if df_all_feats is None else df_all_feats.merge(df_feat, on="molecule", how="outer")

# === Final merge ===
df_final = df_merged.merge(df_all_feats, on="molecule", how="left")
solvent_array = np.vstack(df_final['solvent_onehot'])
solvent_df = pd.DataFrame(solvent_array, columns=[f"solvent_{s}" for s in PREDEFINED_SOLVENTS])
df_final = pd.concat([df_final.drop(columns=['solvent_onehot']).reset_index(drop=True), solvent_df], axis=1)

# === Build X, y ===
drop_cols = ['stimulus','molecule']
if 'SMILES' in df_final.columns: drop_cols.append('SMILES')
df_final = df_final.drop(columns=drop_cols)
feature_columns = [c for c in df_final if c not in label_columns]
X = df_final[feature_columns]
y = df_final[label_columns]

# Save feature list
with open(os.path.join(BASE_MODEL_DIR, "features_used.txt"), "w") as f:
    f.write("\n".join(feature_columns))

# === Train & assemble ===
DEPTHS = [4, 6]
for label in label_columns:
    print(f"\n📈 Training models for: {label}")
    estimators = []
    for d in DEPTHS:
        cb = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=d,
            random_seed=42,
            verbose=100
        )
        cb.fit(X, y[label])
        model_name = f"model_{label}_depth{d}.pkl"
        joblib.dump(cb, os.path.join(BASE_MODEL_DIR, model_name))
        estimators.append((f"cb_depth{d}", cb))
        print(f"✅ Saved {model_name}")

    # build a simple averaging ensemble
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X, y[label])
    ens_name = f"ensemble_{label}.pkl"
    joblib.dump(ensemble, os.path.join(BASE_MODEL_DIR, ens_name))
    print(f"🎉 Saved ensemble model: {ens_name}")

