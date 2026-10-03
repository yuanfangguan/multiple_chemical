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
MODEL_DIR = f"models/{MODEL_NAME}/combined"
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# Remove problematic stimuli if needed
df_test = df_test[~df_test['stimulus'].isin(['AN873'])]

# Build component mappings
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Step 1: Merge chemical features ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for {path}, using latin1")
        df = pd.read_csv(path, encoding='latin1')

    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])

    df.set_index('molecule', inplace=True)
    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient='index')
df_chem_feats.index.name = 'molecule'
df_chem_feats = df_chem_feats.fillna(0)
print(f"✅ Loaded combined chemical feature matrix: {df_chem_feats.shape}")

# === Step 2: Stimulus -> (cid, dilution) pairs ===
def get_pairs(components_str):
    comps = str(components_str).split(';')
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
for _, r in df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])].iterrows():
    pairs = get_pairs(r['components'])
    if pairs:
        stim2pairs[r['id']] = pairs

# === Step 3: Build averaged + max-pooled test features ===
fp_dim = df_chem_feats.shape[1]
cols = (#[f"avg_fp_{i}" for i in range(fp_dim)] +
        [f"max_fp_{i}" for i in range(fp_dim)] +
        ["avg_dilution", "num_chems"])

X_list = []
stim_ids = []
stim_num_components = []  # track component count

for _, row in df_test.iterrows():
    stim = row['stimulus']
    if stim not in stim2pairs:
        print(f"⚠️ No features for stimulus {stim}")
        continue

    pairs = stim2pairs[stim]
    vecs, dils = [], []
    for cid, d in pairs:
        vecs.append(df_chem_feats.loc[cid].values.astype(np.float64))
        dils.append(d)

    if vecs:
        stack = np.vstack(vecs)
        avg_fp = stack.mean(axis=0)
        max_fp = stack.max(axis=0)
        avg_dil = np.mean(dils)
        n_chems = len(dils)
    else:
        avg_fp = np.zeros(fp_dim)
        max_fp = np.zeros(fp_dim)
        avg_dil, n_chems = 0.0, 0

    feat = np.concatenate([max_fp, [avg_dil, n_chems]])
    X_list.append(feat)
    stim_ids.append(stim)
    stim_num_components.append(n_chems)

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)
print(f"✅ Built feature matrix for {len(stim_ids)} stimuli, shape: {X_all.shape}")

# === Step 4: Load models & predict ===
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_', '').replace('.pkl', '')
        model_path = os.path.join(MODEL_DIR, fname)
        print(f"Predicting {label} using {model_path} ...")
        model = joblib.load(model_path)
        final_preds[label] = model.predict(X_all)

# === Step 5: Save predictions ===
df_out = pd.DataFrame({
    'stimulus': stim_ids,
    'num_components': stim_num_components  # record number of components
})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved chemical-only predictions (with num_components) to {OUTPUT_CSV}")

