import pandas as pd
import numpy as np
import os
import joblib

# === File paths ===
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
MODEL_DIR = f"models/{MODEL_NAME}/gated"
GATE_MODEL_PATH = f"models/{MODEL_NAME}/gating_network.pkl"
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

df_test = df_test[~df_test["stimulus"].isin(["AN873"])]

# === Component mappings ===
component_to_cid = dict(zip(df_comp_map["id"], df_comp_map["CID"]))
component_to_dilution = dict(zip(df_comp_map["id"], df_comp_map["dilution"]))

# === Load features ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading: {name}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    if "SMILES" in df.columns:
        df = df.drop(columns=["SMILES"])
    df = df.set_index("molecule")

    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient="index").fillna(0)
df_chem_feats.index.name = "molecule"
print(f"Feature matrix: {df_chem_feats.shape}")

# === Load gating model ===
if not os.path.exists(GATE_MODEL_PATH):
    raise FileNotFoundError(f"Gating model not found: {GATE_MODEL_PATH}")
gate_model = joblib.load(GATE_MODEL_PATH)

# === Gated pooling ===
def gated_pooling(arr, gate_model):
    gates = gate_model.predict(arr)
    gates = np.clip(gates, 0, None)
    if np.sum(gates) < 1e-6:
        w = np.ones_like(gates) / len(gates)
    else:
        w = gates / np.sum(gates)
    return np.sum(w[:, None] * arr, axis=0)

# === Stimulus → pairs ===
def get_pairs(components_str):
    out = []
    for s in str(components_str).split(";"):
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid in df_chem_feats.index and dil is not None:
                out.append((cid, float(dil)))
    return out

stim2pairs = {
    r["id"]: get_pairs(r["components"])
    for _, r in df_stim_map[df_stim_map["id"].isin(df_test["stimulus"])].iterrows()
}

# === Build test features ===
fp_dim = df_chem_feats.shape[1]
cols = [f"gated_fp_{i}" for i in range(fp_dim)] + ["avg_dilution", "num_chems"]

X_list = []
stim_ids = []
num_comp_list = []   # <<< ADDED

for _, r in df_test.iterrows():
    sid = r["stimulus"]
    pairs = stim2pairs.get(sid, [])
    vecs, dils = [], []

    for cid, d in pairs:
        vecs.append(df_chem_feats.loc[cid].values.astype(float))
        dils.append(d)

    if vecs:
        arr = np.vstack(vecs)
        pooled = gated_pooling(arr, gate_model)
        avg_dil = float(np.mean(dils))
        n_chems = len(dils)
    else:
        pooled = np.zeros(fp_dim)
        avg_dil = 0.0
        n_chems = 0

    feat = np.concatenate([pooled, [avg_dil, n_chems]])
    X_list.append(feat)
    stim_ids.append(sid)
    num_comp_list.append(n_chems)   # <<< ADDED

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)

# === Predict ===
final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith(".pkl"):
        label = fname.replace("model_","").replace(".pkl","")
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        final_preds[label] = model.predict(X_all)

# === Output CSV ===
df_out = pd.DataFrame({"stimulus": stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

df_out["num_components"] = num_comp_list   # <<< ADDED

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n🎉 Saved predictions with num_components → {OUTPUT_CSV}")

