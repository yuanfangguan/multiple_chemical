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

MODEL_NAME = "combined_with_rata"
MODEL_DIR = f"models/{MODEL_NAME}/combined"
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)
df_rata_single = pd.read_csv(SINGLE_RATA_PATH)

# Optional: remove problematic stimuli
df_test = df_test[~df_test['stimulus'].isin(['AN873'])]

# === Build component mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Step 1: Prepare RATA-only features ===
common = df_rata_single.merge(df_cid, on='molecule', how='inner').drop_duplicates('molecule')
rata_cols = [c for c in common.columns if c not in ('SMILES', 'molecule', 'stimulus', 'components', 'dilution')]
common[rata_cols] = common[rata_cols].apply(pd.to_numeric, errors='coerce')

df_combined_feats = (
    common.set_index('molecule')[rata_cols]
    .fillna(0)
)

print(f"✅ RATA-only features loaded, shape: {df_combined_feats.shape}")

# === Step 2: Stimulus -> (cid, dilution) mapping ===
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

# === Step 3: Build test feature matrix ===
fp_dim = df_combined_feats.shape[1]
cols = ([f"avg_fp_{i}" for i in range(fp_dim)] +
        [f"max_fp_{i}" for i in range(fp_dim)] +
        ["avg_dilution", "num_chems"])

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
        if cid in df_combined_feats.index:
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
    feat = np.concatenate([avg_fp, max_fp, [avg_dil, n_chems]])
    X_list.append(feat)
    stim_ids.append(stim)

X_all = pd.DataFrame(X_list, columns=cols).fillna(0)
print(f"✅ Built feature matrix for {len(stim_ids)} stimuli, dim={X_all.shape}")

# === Step 4: Load models and predict ===
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"❌ Model directory not found: {MODEL_DIR}")

final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_', '').replace('.pkl', '')
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        print(f"Predicting {label}...")
        final_preds[label] = model.predict(X_all)

# === Step 5: Save predictions ===
df_out = pd.DataFrame({'stimulus': stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved final predictions to {OUTPUT_CSV}")

