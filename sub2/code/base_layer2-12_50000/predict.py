import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# === File paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
FEATURE_DIR = "../../../data_external"  # Features base directory

# === Load shared files ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

# === Prepare: component ID → CID mapping
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))

# === Stimulus to list of CIDs ===
def get_cids(components_str):
    comps = str(components_str).split(';')
    cids = []
    for comp_id in comps:
        comp_id = comp_id.strip()
        if comp_id.isdigit():
            cid = component_to_cid.get(int(comp_id))
            if cid is not None:
                cids.append(cid)
    return cids

# === Build stimulus → CIDs dict for test set
stimulus_to_cids = {}
df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]

for _, row in df_stim_map_filtered.iterrows():
    stimulus = row['id']
    cids = get_cids(row['components'])
    if cids:
        stimulus_to_cids[stimulus] = cids

# === Predict for layers 2 to 12 ===
for layer in range(2, 13):
    print(f"\n🔍 Predicting using features from layer {layer}")
    FEATURES_PATH = os.path.join(FEATURE_DIR, f"50000_sub2_cid_smiles_hidden_features_layer{layer}.csv")
    MODEL_DIR = f"models_layer{layer}"
    OUTPUT_CSV = f"predictions_layer{layer}.csv"

    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory {MODEL_DIR} not found, skipping layer {layer}")
        continue

    df_feats = pd.read_csv(FEATURES_PATH)
    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    # Calculate max_cids for this layer
    max_cids = max(len([cid for cid in cids if cid in cid_to_feats.index]) for cids in stimulus_to_cids.values())

    print(f"📌 Max components in test for layer {layer}: {max_cids}")
    print(f"📌 Fingerprint dimension per CID: {fingerprint_dim}")

    # === Build final test feature matrix ===
    stimulus_ids = []
    feature_matrix = []

    for _, row in df_test.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_cids:
            cids = [cid for cid in stimulus_to_cids[stim] if cid in cid_to_feats.index]
            if not cids:
                print(f"⚠️ No features found for stimulus: {stim} in layer {layer}")
                continue

            feat_vecs = [cid_to_feats.loc[cid].values for cid in cids]

            while len(feat_vecs) < max_cids:
                feat_vecs.append(np.zeros(fingerprint_dim))

            concat_feat = np.concatenate(feat_vecs)
            feature_matrix.append(concat_feat)
            stimulus_ids.append(stim)
        else:
            print(f"⚠️ No mapping found for stimulus: {stim}")

    if not feature_matrix:
        print(f"❌ No valid test samples for layer {layer}, skipping.")
        continue

    X_test = pd.DataFrame(feature_matrix)
    predictions = pd.DataFrame({'stimulus': stimulus_ids})

    # === Predict using each model for this layer ===
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
    print(f"✅ Predictions for layer {layer} saved to {OUTPUT_CSV}")

print("\n✅ All layers processed.")

