import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# === File paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
FEATURES_PATH = "../../../data_external/sub2_cid_smiles_hidden_features_layer5.csv"
MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_feats = pd.read_csv(FEATURES_PATH)

# Drop SMILES if exists
if 'SMILES' in df_feats.columns:
    df_feats = df_feats.drop(columns=['SMILES'])

# === Prepare: component ID → CID mapping
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))

# === CID → fingerprint vector
cid_to_feats = df_feats.set_index('molecule')
fingerprint_dim = cid_to_feats.shape[1]

# === Stimulus to list of CIDs ===
def get_cids(components_str):
    comps = str(components_str).split(';')
    cids = []
    for comp_id in comps:
        comp_id = comp_id.strip()
        if comp_id.isdigit():
            cid = component_to_cid.get(int(comp_id))
            if cid is not None and cid in cid_to_feats.index:
                cids.append(cid)
    return cids

# === Build stimulus → CIDs dict for test set
stimulus_to_cids = {}
max_cids = 0

df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]

for _, row in df_stim_map_filtered.iterrows():
    stimulus = row['id']
    cids = get_cids(row['components'])
    if cids:
        stimulus_to_cids[stimulus] = cids
        max_cids = max(max_cids, len(cids))

print(f"📌 Max components in test: {max_cids}")
print(f"📌 Fingerprint dimension per CID: {fingerprint_dim}")

# === Build final test feature matrix ===
stimulus_ids = []
feature_matrix = []

for _, row in df_test.iterrows():
    stim = row['stimulus']
    if stim in stimulus_to_cids:
        cids = stimulus_to_cids[stim]
        feat_vecs = []
        for cid in cids:
            feat_vecs.append(cid_to_feats.loc[cid].values)

        # Pad with zeros to max_cids
        while len(feat_vecs) < max_cids:
            feat_vecs.append(np.zeros(fingerprint_dim))

        concat_feat = np.concatenate(feat_vecs)
        feature_matrix.append(concat_feat)
        stimulus_ids.append(stim)
    else:
        print(f"⚠️ No feature found for stimulus: {stim}")

X_test = pd.DataFrame(feature_matrix)

# === Predict using each model ===
predictions = pd.DataFrame({'stimulus': stimulus_ids})

for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        label = model_file.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"🔍 Loading model: {model_path}")
        
        model = joblib.load(model_path)
        preds = model.predict(X_test)
        predictions[label] = preds

# === Save predictions ===
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")

