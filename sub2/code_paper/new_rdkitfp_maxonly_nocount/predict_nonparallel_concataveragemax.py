import pandas as pd
import os
import joblib
import numpy as np

# === File paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
SINGLE_RATA_PATH = "../../data/raw/Task2_single_RATA.csv"

FEATURE_SET_PATHS = {
#    "maccs": "../../data/processed/features_maccs.csv",
#    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
#    "descriptors": "../../data/processed/features_descriptors.csv",
#    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}

MODEL_NAME = "combined_with_rata"
MODEL_DIR = f"models/{MODEL_NAME}/combined"
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)
df_rata_single = pd.read_csv(SINGLE_RATA_PATH)

# Remove any problematic stimuli if needed
df_test = df_test[~df_test['stimulus'].isin(['AN873'])]

# Build component mappings
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Step 1: Merge chemical features ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
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

# === Step 2: Merge with single-RATA ===
common = df_rata_single.merge(df_cid, on='molecule', how='inner').drop_duplicates('molecule')
rata_cols = [c for c in common.columns if c not in ('SMILES','molecule','stimulus','components','dilution')]
common[rata_cols] = common[rata_cols].apply(pd.to_numeric, errors='coerce')

df_combined_feats = (
    df_chem_feats
    .join(common.set_index('molecule')[rata_cols], how='left')
    .fillna(0)
)

# === Step 3: Stimulus -> (cid,dilution) pairs ===
def get_pairs(components_str):
    comps = str(components_str).split(';')
    out = []
    for s in comps:
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid in df_combined_feats.index and dil is not None:
                out.append((cid, float(dil)))
    return out

stim2pairs = {}
for _, r in df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])].iterrows():
    pairs = get_pairs(r['components'])
    if pairs:
        stim2pairs[r['id']] = pairs

# === Step 4: Build combined test features ===
fp_dim = df_combined_feats.shape[1]
cols = (
        [f"max_fp_{i}" for i in range(fp_dim)] +
        ["avg_dilution"])

X_list = []
stim_ids = []
for _, row in df_test.iterrows():
    stim = row['stimulus']
    if stim not in stim2pairs:
        print(f"⚠️ No features for stimulus {stim}")
        continue
    pairs = stim2pairs[stim]
    vecs, dils = [], []
    for cid, d in pairs:
        vecs.append(df_combined_feats.loc[cid].values.astype(np.float64))
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
    feat = np.concatenate([max_fp, [avg_dil]])
    X_list.append(feat)
    stim_ids.append(stim)

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)

# === Step 5: Load & predict with combined models ===
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_','').replace('.pkl','')
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        print(f"Predicting {label}...")
        final_preds[label] = model.predict(X_all)

# === Step 6: Save output ===
df_out = pd.DataFrame({'stimulus': stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved final predictions to {OUTPUT_CSV}")

