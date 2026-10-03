import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

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
component_to_cid = dict(zip(df_comp_map["id"], df_comp_map["CID"]))
component_to_dilution = dict(zip(df_comp_map["id"], df_comp_map["dilution"]))

# === Merge all chemical fingerprint feature sets ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    if "SMILES" in df.columns:
        df = df.drop(columns=["SMILES"])

    df = df.set_index("molecule")
    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient="index")
df_chem_feats.index.name = "molecule"
df_chem_feats = df_chem_feats.fillna(0).reset_index()
print(f"✅ Combined chemical feature matrix: {df_chem_feats.shape}")

# === Build mapping for each stimulus ===
def get_pairs(components_str):
    comps = str(components_str).split(";")
    out = []
    for s in comps:
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid in df_chem_feats["molecule"].values and dil is not None:
                out.append((cid, float(dil)))
    return out

stim2pairs = {}
for _, r in df_stim_map.iterrows():
    pairs = get_pairs(r["components"])
    if pairs:
        stim2pairs[r["id"]] = pairs

# === Define nonlinear pooling ===
def nonlinear_pool(arr, model):
    """arr: (n_chems, n_feats) -> fused vector"""
    feat_mean = arr.mean(axis=0)
    feat_max = arr.max(axis=0)
    feat_min = arr.min(axis=0)
    pooled_stats = np.concatenate([feat_mean, feat_max, feat_min])[None, :]
    fused_vec = model.predict(pooled_stats)[0]  # shape: (output_dim,)
    return fused_vec

# === Main processing and training ===
def process_feature_set(name, df_feats):
    print(f"\n=== Processing chemical feature set: {name} (NONLINEAR POOLING) ===")
    df_feats = df_feats.set_index("molecule")
    fp_dim = df_feats.shape[1]

    # === Train small nonlinear pooling MLP ===
    print("🔧 Training nonlinear pooling network...")
    X_unsup, Y_unsup = [], []
    for _ in range(3000):
        cids = np.random.choice(df_feats.index, size=np.random.randint(2, 10), replace=False)
        arr = df_feats.loc[cids].values
        feat_mean = arr.mean(axis=0)
        feat_max = arr.max(axis=0)
        feat_min = arr.min(axis=0)
        pooled_stats = np.concatenate([feat_mean, feat_max, feat_min])
        X_unsup.append(pooled_stats)
        Y_unsup.append(feat_mean)  # target = mean features (vector!)
    X_unsup = np.array(X_unsup)
    Y_unsup = np.array(Y_unsup)
    nonlinear_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
    )
    nonlinear_model.fit(X_unsup, Y_unsup)
    print(f"✅ Nonlinear pooling network trained. Input {X_unsup.shape[1]} → Output {Y_unsup.shape[1]}")

    # === Aggregate mixture features ===
    X_list, rows = [], []
    for _, r in tqdm(df_train.iterrows(), total=len(df_train)):
        sid = r["stimulus"]
        if sid not in stim2pairs:
            continue
        vecs, dils = [], []
        for cid, d in stim2pairs[sid]:
            if cid in df_feats.index:
                vecs.append(df_feats.loc[cid].values.astype(float))
                dils.append(d)
        if not vecs:
            continue
        arr = np.vstack(vecs)
        fused_fp = nonlinear_pool(arr, nonlinear_model)
        avg_dil = np.mean(dils)
        n_chems = len(dils)
        feat = np.concatenate([fused_fp, [avg_dil, n_chems]])
        X_list.append(feat)
        rows.append(r)

    df_usable = pd.DataFrame(rows)
    y = df_usable.drop(columns=["stimulus"])
    X_all = pd.DataFrame(np.array(X_list),
                         columns=[f"nlp_fp_{i}" for i in range(fp_dim)] + ["avg_dilution", "num_chems"])

    # === Train CatBoost per target ===
    def train_and_save(label):
        out_dir = os.path.join(OUTPUT_MODEL_DIR, name, "nonlinear")
        os.makedirs(out_dir, exist_ok=True)
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            random_seed=42,
            verbose=100
        )
        Xc = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)
        print(f"Training {label} -> {out_dir}")
        model.fit(Xc, y[label])
        joblib.dump(model, os.path.join(out_dir, f"model_{label}.pkl"))
        print(f"✅ Saved model_{label}.pkl")

    Parallel(n_jobs=-1)(delayed(train_and_save)(lbl) for lbl in y.columns)
    joblib.dump(nonlinear_model, os.path.join(OUTPUT_MODEL_DIR, name, "nonlinear_pooling_network.pkl"))
    print(f"\n✅ All models and nonlinear pooling network saved under {OUTPUT_MODEL_DIR}/{name}/nonlinear/")

# === Execute ===
process_feature_set("combined_no_rata", df_chem_feats)
print("\n✅ Done (Nonlinear pooling).")

