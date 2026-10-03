import pandas as pd
import os
import joblib
import numpy as np

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
MODEL_DIR = f"models/{MODEL_NAME}/softmax_pool"   # <-- unchanged
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

df_test = df_test[~df_test["stimulus"].isin(["AN873"])]

# === Build mappings ===
component_to_cid = dict(zip(df_comp_map["id"], df_comp_map["CID"]))
component_to_dilution = dict(zip(df_comp_map["id"], df_comp_map["dilution"]))

# === Merge chemical feature sets ===
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
df_chem_feats = df_chem_feats.fillna(0)

print(f"✅ Loaded combined chemical feature matrix: {df_chem_feats.shape}")

# === Build Stimulus → component pairs ===
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

# === Pooling ===
fp_dim = df_chem_feats.shape[1]
cols = ([f"softmax_fp_{i}" for i in range(fp_dim)] +
        ["avg_dilution", "num_chems"])

alpha = 1.0

X_list = []
stim_ids = []
num_comp_list = []   # <<<< new list to store num_components

for _, row in df_test.iterrows():
    stim = row["stimulus"]
    if stim not in stim2pairs:
        print(f"⚠️ No features for stimulus {stim}")
        continue

    pairs = stim2pairs[stim]
    vecs, scores = [], []

    for cid, d in pairs:
        vecs.append(df_chem_feats.loc[cid].values.astype(np.float64))
        scores.append(float(d))

    if len(vecs) == 0:
        pooled = np.zeros(fp_dim)
        avg_dil = 0.0
        n_chems = 0
    else:
        arr = np.vstack(vecs)
        sc = np.array(scores)

        exp_scores = np.exp(alpha * sc)
        w = exp_scores / (exp_scores.sum() + 1e-9)

        pooled = (w[:, None] * np.tanh(arr)).sum(axis=0)

        avg_dil = float(sc.mean())
        n_chems = len(sc)

    feat = np.concatenate([pooled, [avg_dil, n_chems]])
    X_list.append(feat)
    stim_ids.append(stim)
    num_comp_list.append(n_chems)    # <<<< save number of components

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)

# === Load models and predict ===
final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith(".pkl"):
        label = fname.replace("model_","").replace(".pkl","")
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        final_preds[label] = model.predict(X_all)

# === Build output ===
df_out = pd.DataFrame({"stimulus": stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

# ADD THIS COLUMN
df_out["num_components"] = num_comp_list   # <<<< new

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n🎉 Saved predictions with num_components to {OUTPUT_CSV}")

