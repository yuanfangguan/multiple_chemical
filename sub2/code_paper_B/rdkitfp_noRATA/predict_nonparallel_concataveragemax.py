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
#    "maccs": "../../data/processed/features_maccs.csv",
#    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
#    "descriptors": "../../data/processed/features_descriptors.csv",
#    "mordred": "../../data/raw/Mordred_Descriptors.csv"
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

# Filter any missing or invalid stimuli
df_test = df_test[~df_test["stimulus"].isin(["AN873"])]

# === Build component mappings ===
component_to_cid = dict(zip(df_comp_map["id"], df_comp_map["CID"]))
component_to_dilution = dict(zip(df_comp_map["id"], df_comp_map["dilution"]))

# === Step 1: Load chemical feature sets ===
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

# === Step 2: Load the trained gating network ===
if not os.path.exists(GATE_MODEL_PATH):
    raise FileNotFoundError(f"Gating model not found: {GATE_MODEL_PATH}")

gate_model = joblib.load(GATE_MODEL_PATH)
print(f"✅ Loaded gating model from {GATE_MODEL_PATH}")

# === Step 3: Define gated pooling ===
def gated_pooling(arr, gate_model):
    gates = gate_model.predict(arr)
    gates = np.clip(gates, 0, None)
    if np.sum(gates) < 1e-6:
        weights = np.ones_like(gates) / len(gates)
    else:
        weights = gates / np.sum(gates)
    pooled = np.sum(weights[:, None] * arr, axis=0)
    return pooled

# === Step 4: Build Stimulus → (CID, dilution) map ===
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

stim2pairs = {}
for _, r in df_stim_map[df_stim_map["id"].isin(df_test["stimulus"])].iterrows():
    pairs = get_pairs(r["components"])
    if pairs:
        stim2pairs[r["id"]] = pairs

# === Step 5: Aggregate test features using the gating model ===
fp_dim = df_chem_feats.shape[1]
cols = [f"gated_fp_{i}" for i in range(fp_dim)] + ["avg_dilution", "num_chems"]

X_list, stim_ids = [], []
for _, row in df_test.iterrows():
    sid = row["stimulus"]
    if sid not in stim2pairs:
        print(f"⚠️ Missing components for {sid}")
        continue
    pairs = stim2pairs[sid]
    vecs, dils = [], []
    for cid, d in pairs:
        vecs.append(df_chem_feats.loc[cid].values.astype(np.float64))
        dils.append(d)

    if vecs:
        arr = np.vstack(vecs)
        gated_fp = gated_pooling(arr, gate_model)
        avg_dil = np.mean(dils)
        n_chems = len(dils)
    else:
        gated_fp = np.zeros(fp_dim)
        avg_dil = 0.0
        n_chems = 0
    feat = np.concatenate([gated_fp, [avg_dil, n_chems]])
    X_list.append(feat)
    stim_ids.append(sid)

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)
print(f"✅ Built gated-pooled test feature matrix: {X_all.shape}")

# === Step 6: Load CatBoost models and predict ===
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith(".pkl"):
        label = fname.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(MODEL_DIR, fname)
        print(f"🔮 Predicting {label} using {model_path} ...")
        model = joblib.load(model_path)
        final_preds[label] = model.predict(X_all)

# === Step 7: Save predictions ===
df_out = pd.DataFrame({"stimulus": stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved gated-pooling predictions to {OUTPUT_CSV}")

