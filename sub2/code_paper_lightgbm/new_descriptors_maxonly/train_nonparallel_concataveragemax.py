import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from joblib import Parallel, delayed

# Clean up old models
os.system("rm -rf models*")

# === Paths ===
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
FEATURE_SET_PATHS = {
    "descriptors": "../../data/processed/features_descriptors.csv",
}
OUTPUT_MODEL_DIR = "models"

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

# === Build component maps ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Merge chemical feature sets ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')

    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])

    # --- Auto-detect ID column ---
    if 'molecule' in df.columns:
        id_col = 'molecule'
    elif 'CID' in df.columns:
        id_col = 'CID'
    elif 'id' in df.columns:
        id_col = 'id'
    else:
        id_col = df.columns[0]  # assume first column is ID

    df = df.set_index(id_col)
    print(f"→ Using '{id_col}' as ID column for {name}")

    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

# Create DataFrame of merged chemical features
df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient='index')
df_chem_feats.index.name = 'molecule'

print("\n✅ Chemical features loaded and merged.")
print(f"Shape: {df_chem_feats.shape}")
print(df_chem_feats.head())


# === Define and run feature processing & training ===
def process_feature_set(name, df_feats):
    print(f"\n=== Processing feature set: {name} ===")

    # ✅ Fix: don’t reset index if already set
    if 'molecule' in df_feats.columns:
        df_feats = df_feats.set_index('molecule')

    fp_dim = df_feats.shape[1]

    # Build stimulus -> [(cid, dilution)] mapping
    def get_pairs(components_str):
        ids = str(components_str).split(';')
        out = []
        for s in ids:
            if s.strip().isdigit():
                comp = int(s)
                cid = component_to_cid.get(comp)
                dil = component_to_dilution.get(comp)
                if cid in df_feats.index and dil is not None:
                    out.append((cid, float(dil)))
        return out

    stim2pairs = {}
    for _, r in df_stim_map.iterrows():
        pairs = get_pairs(r['components'])
        if pairs:
            stim2pairs[r['id']] = pairs

    X_list, rows = [], []
    for _, r in df_train.iterrows():
        sid = r['stimulus']
        if sid in stim2pairs:
            vecs, dils = [], []
            for cid, d in stim2pairs[sid]:
                if cid in df_feats.index:
                    vecs.append(df_feats.loc[cid].values.astype(float))
                    dils.append(d)
            if vecs:
                arr = np.vstack(vecs)
                avg_fp = arr.mean(axis=0)
                max_fp = arr.max(axis=0)
                avg_dil = np.mean(dils)
                n_chems = len(dils)
            else:
                avg_fp = np.zeros(fp_dim)
                max_fp = np.zeros(fp_dim)
                avg_dil = 0.0
                n_chems = 0
            feat = np.concatenate([max_fp, [avg_dil, n_chems]])
            X_list.append(feat)
            rows.append(r)

    df_usable = pd.DataFrame(rows)
    y = df_usable.drop(columns=['stimulus'])

    # Build feature DataFrame
    cols = (#[f"avg_fp_{i}" for i in range(fp_dim)] +
            [f"max_fp_{i}" for i in range(fp_dim)] +
            ["avg_dilution", "num_chems"])
    X_all = pd.DataFrame(np.array(X_list), columns=cols)

    # Train one model per label
    def train_and_save(label):
        out_dir = os.path.join(OUTPUT_MODEL_DIR, name, 'chemical_only')
        os.makedirs(out_dir, exist_ok=True)
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
        )
        Xc = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)
        print(f"Training {label} -> {out_dir}")
        model.fit(Xc, y[label])
        joblib.dump(model, os.path.join(out_dir, f"model_{label}.pkl"))
        print(f"✅ Saved model_{label}.pkl")

    Parallel(n_jobs=-1)(delayed(train_and_save)(lbl) for lbl in y.columns)
    print(f"\n✅ All models saved under {OUTPUT_MODEL_DIR}/{name}/chemical_only/")

# === Execute ===
process_feature_set("descriptors_only", df_chem_feats)
print("\n✅ Done.")

