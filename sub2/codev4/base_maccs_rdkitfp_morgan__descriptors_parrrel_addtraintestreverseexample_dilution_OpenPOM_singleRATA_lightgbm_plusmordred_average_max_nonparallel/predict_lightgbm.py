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
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}

MODEL_NAME = "combined_with_rata"
MODEL_DIR = f"models/{MODEL_NAME}"
OUTPUT_CSV = "predictions.csv"

# === Load base data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)
df_rata_single = pd.read_csv(SINGLE_RATA_PATH)

# === Filter out problematic stimuli if needed ===
df_test = df_test[~df_test['stimulus'].isin(['AN873'])]

# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Step 1: Merge chemical features ===
merged_feats_dict = {}
for name, path in FEATURE_SET_PATHS.items():
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for {path}, using latin1")
        df = pd.read_csv(path, encoding='latin1')
    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])
    df.set_index('molecule', inplace=True)

    for cid in df.index:
        if cid not in merged_feats_dict:
            merged_feats_dict[cid] = []
        merged_feats_dict[cid].extend(df.loc[cid].values.tolist())

merged_cids = list(merged_feats_dict.keys())
merged_vectors = [merged_feats_dict[cid] for cid in merged_cids]
df_chem_feats = pd.DataFrame(merged_vectors, index=merged_cids)
df_chem_feats.index.name = 'molecule'

# === Step 2: Merge with single RATA ===
df_rata_single_mapped = df_rata_single.merge(df_cid, on='molecule', how='inner').drop_duplicates(subset='molecule')
rata_cols = [col for col in df_rata_single_mapped.columns if col not in ('SMILES', 'molecule', 'stimulus', 'components', 'dilution')]
df_rata_single_mapped[rata_cols] = df_rata_single_mapped[rata_cols].apply(pd.to_numeric, errors='coerce')
df_combined_feats = df_chem_feats.join(df_rata_single_mapped.set_index('molecule')[rata_cols], how='left').fillna(0)

cid_to_feats = df_combined_feats
fingerprint_dim = cid_to_feats.shape[1]

# === Step 3: Stimulus → (CID, dilution) pairs ===
def get_cid_dilution_pairs(components_str):
    comps = str(components_str).split(';')
    pairs = []
    for comp_str in comps:
        if comp_str.strip().isdigit():
            comp_id = int(comp_str.strip())
            cid = component_to_cid.get(comp_id)
            dilution = component_to_dilution.get(comp_id)
            if cid is not None and dilution is not None:
                pairs.append((cid, float(dilution)))
    return pairs

stimulus_to_pairs = {}
df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]
for _, row in df_stim_map_filtered.iterrows():
    stim = row['id']
    pairs = [(cid, dilution) for cid, dilution in get_cid_dilution_pairs(row['components']) if cid in cid_to_feats.index]
    if pairs:
        stimulus_to_pairs[stim] = pairs

# === Step 4: Construct test feature matrix ===
stimulus_ids = []
X_avg, X_max = [], []

def build_vector(pairs):
    vecs = []
    dilutions = []
    for cid, dilution in pairs:
        feats = cid_to_feats.loc[cid].values.astype(np.float64)
        vecs.append(feats)
        dilutions.append(dilution)
    if not vecs:
        return np.zeros(fingerprint_dim + 2), np.zeros(fingerprint_dim + 2)
    stacked = np.vstack(vecs)
    avg = np.mean(stacked, axis=0)
    maxv = np.max(stacked, axis=0)
    avg_dil = np.mean(dilutions)
    num_chems = len(pairs)
    return np.concatenate([avg, [avg_dil, num_chems]]), np.concatenate([maxv, [avg_dil, num_chems]])

for _, row in df_test.iterrows():
    stim = row['stimulus']
    if stim in stimulus_to_pairs:
        avg_vec, max_vec = build_vector(stimulus_to_pairs[stim])
        X_avg.append(avg_vec)
        X_max.append(max_vec)
        stimulus_ids.append(stim)
    else:
        print(f"⚠️ No features for stimulus {stim}")

X_avg = pd.DataFrame(X_avg).fillna(0)
X_max = pd.DataFrame(X_max).fillna(0)

# === Step 5: Predict using combined model ===
final_preds = {}

for version, X in [("avg", X_avg), ("max", X_max)]:
    model_path = os.path.join(MODEL_DIR, version)
    if not os.path.exists(model_path):
        print(f"⚠️ Missing model directory: {model_path}")
        continue
    for fname in os.listdir(model_path):
        if fname.endswith(".pkl"):
            label = fname.replace("model_", "").replace(".pkl", "")
            print(f"🔍 Predicting {label} [{version}]")
            model = joblib.load(os.path.join(model_path, fname))
            preds = model.predict(X)
            if label not in final_preds:
                final_preds[label] = preds
            else:
                final_preds[label] += preds

# === Average predictions
for label in final_preds:
    final_preds[label] /= 2  # avg + max

# === Save
df_out = pd.DataFrame({'stimulus': stimulus_ids})
for label in sorted(final_preds.keys()):
    df_out[label] = final_preds[label]

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved final predictions to {OUTPUT_CSV}")

