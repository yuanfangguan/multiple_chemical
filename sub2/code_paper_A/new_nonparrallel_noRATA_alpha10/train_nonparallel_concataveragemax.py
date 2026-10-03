import pandas as pd
import numpy as np
import os
import joblib
from catboost import CatBoostRegressor
from joblib import Parallel, delayed

# Clean up old models
os.system("rm -rf models*")

# === Paths ===
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}
OUTPUT_MODEL_DIR = "models"

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Build component maps ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Merge all chemical fingerprint feature sets ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')

    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])

    df = df.set_index('molecule')

    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

# Create DataFrame of merged chemical features
df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient='index')
df_chem_feats.index.name = 'molecule'
df_chem_feats = df_chem_feats.fillna(0).reset_index()

print(f"✅ Combined chemical feature matrix: {df_chem_feats.shape}")

# === Define & run training ===
def process_feature_set(name, df_feats):
    print(f"\n=== Processing chemical feature set: {name} ===")
    df_feats = df_feats.set_index('molecule')
    fp_dim = df_feats.shape[1]

    # Build stimulus → [(CID, dilution)] map
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

    # === Softmax-weighted non-linear pooling ===
    X_list, rows = [], []
    alpha = 10.0  # temperature hyperparameter

    for _, r in df_train.iterrows():
        sid = r['stimulus']
        if sid not in stim2pairs:
            continue

        vecs, scores = [], []
        for cid, d in stim2pairs[sid]:
            if cid in df_feats.index:
                v = df_feats.loc[cid].values.astype(float)
                vecs.append(v)
                scores.append(float(d))

        if len(vecs) == 0:
            pooled = np.zeros(fp_dim)
            avg_dil = 0.0
            n_chems = 0
        else:
            arr = np.vstack(vecs)        # [n, fp_dim]
            sc = np.array(scores)        # [n]

            # softmax weights over dilution score
            exp_scores = np.exp(alpha * sc)
            w = exp_scores / (exp_scores.sum() + 1e-9)  # [n]

            # non-linear transformation + weighted pooling
            pooled = (w[:, None] * np.tanh(arr)).sum(axis=0)  # [fp_dim]

            avg_dil = sc.mean()
            n_chems = len(sc)

        feat = np.concatenate([
            pooled,
            [avg_dil, n_chems]
        ])

        X_list.append(feat)
        rows.append(r)

    # Prepare target and input matrices
    df_usable = pd.DataFrame(rows)
    y = df_usable.drop(columns=['stimulus'])

    # feature names changed
    cols = (
        [f"softmax_fp_{i}" for i in range(fp_dim)] +
        ["avg_dilution", "num_chems"]
    )

    X_all = pd.DataFrame(np.array(X_list), columns=cols)

    # === Train and save models per label ===
    def train_and_save(label):
        out_dir = os.path.join(OUTPUT_MODEL_DIR, name, 'softmax_pool')
        os.makedirs(out_dir, exist_ok=True)
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            random_seed=42,
            verbose=100
        )
        Xc = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)
        print(f"Training {label} -> {out_dir}")
        model.fit(Xc, y[label])
        joblib.dump(model, os.path.join(out_dir, f"model_{label}.pkl"))
        print(f"✅ Saved model_{label}.pkl")

    Parallel(n_jobs=-1)(delayed(train_and_save)(lbl) for lbl in y.columns)
    print(f"\n✅ All models saved under {OUTPUT_MODEL_DIR}/{name}/softmax_pool/")

# === Execute ===
process_feature_set("combined_no_rata", df_chem_feats)
print("\n✅ Done (softmax-weighted non-linear pooling).")

