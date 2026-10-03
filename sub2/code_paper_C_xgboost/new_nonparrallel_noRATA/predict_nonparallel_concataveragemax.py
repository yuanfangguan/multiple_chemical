import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm

# === Paths ===
TEST_PATH = "test.csv"
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

MODEL_NAME = "combined_no_rata"
MODEL_DIR = f"models/{MODEL_NAME}/nonlinear"
POOL_NET_PATH = os.path.join(MODEL_DIR, "../nonlinear_pooling_network.pkl")
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# Remove any problematic stimuli if needed
df_test = df_test[~df_test["stimulus"].isin(["AN873"])]

# === Load trained nonlinear pooling network ===
if not os.path.exists(POOL_NET_PATH):
    raise FileNotFoundError(f"❌ Nonlinear pooling network not found at {POOL_NET_PATH}")
pool_model = joblib.load(POOL_NET_PATH)
print(f"✅ Loaded nonlinear pooling network: {POOL_NET_PATH}")

# === Build component mappings ===
component_to_cid = dict(zip(df_comp_map["id"], df_comp_map["CID"]))
component_to_dilution = dict(zip(df_comp_map["id"], df_comp_map["dilution"]))

# === Merge chemical feature sets ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for {path}, using latin1")
        df = pd.read_csv(path, encoding="latin1")

    if "SMILES" in df.columns:
        df = df.drop(columns=["SMILES"])

    df.set_index("molecule", inplace=True)
    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient="index")
df_chem_feats.index.name = "molecule"
df_chem_feats = df_chem_feats.fillna(0)
print(f"✅ Loaded combined chemical feature matrix: {df_chem_feats.shape}")

# === Helper functions ===
def get_pairs(components_str):
    comps = str(components_str).split(";")
    out = []
    for s in comps:
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid in df_chem_feats.index and dil is not None:
                out.append((cid, float(dil)))
    return out

def nonlinear_pool(arr, model):
    """Perform nonlinear pooling via trained MLP model"""
    feat_mean = arr.mean(axis=0)
    feat_max = arr.max(axis=0)
    feat_min = arr.min(axis=0)
    pooled_stats = np.concatenate([feat_mean, feat_max, feat_min])[None, :]
    fused_vec = model.predict(pooled_stats)[0]
    return fused_vec

# === Build test mapping ===
stim2pairs = {}
for _, r in df_stim_map[df_stim_map["id"].isin(df_test["stimulus"])].iterrows():
    pairs = get_pairs(r["components"])
    if pairs:
        stim2pairs[r["id"]] = pairs

# === Generate pooled test features ===
fp_dim = df_chem_feats.shape[1]
cols = [f"nlp_fp_{i}" for i in range(fp_dim)] + ["avg_dilution", "num_chems"]

X_list, stim_ids = [], []
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    sid = row["stimulus"]
    if sid not in stim2pairs:
        continue
    pairs = stim2pairs[sid]
    vecs, dils = [], []
    for cid, d in pairs:
        if cid in df_chem_feats.index:
            vecs.append(df_chem_feats.loc[cid].values.astype(float))
            dils.append(d)
    if not vecs:
        continue
    arr = np.vstack(vecs)
    fused_fp = nonlinear_pool(arr, pool_model)
    avg_dil = np.mean(dils)
    n_chems = len(dils)
    feat = np.concatenate([fused_fp, [avg_dil, n_chems]])
    X_list.append(feat)
    stim_ids.append(sid)

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)
print(f"✅ Built nonlinear pooled feature matrix for {len(stim_ids)} stimuli.")

# === Load trained CatBoost models and predict ===
final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith(".pkl") and "pooling_network" not in fname:
        label = fname.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(MODEL_DIR, fname)
        print(f"🔮 Predicting {label} using {model_path}")
        model = joblib.load(model_path)
        final_preds[label] = model.predict(X_all)

# === Save predictions ===
df_out = pd.DataFrame({"stimulus": stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved nonlinear pooled predictions to {OUTPUT_CSV}")

