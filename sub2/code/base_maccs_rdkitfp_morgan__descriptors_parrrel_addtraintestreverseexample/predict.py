import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# === File paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
}
MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Load shared data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))

# === Helper: Get CIDs from component string
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

# === Prepare final predictions container ===
stimulus_ids = []
final_preds = {}

# === For each feature set ===
for feat_name, feat_path in FEATURE_SET_PATHS.items():
    print(f"\n=== Processing feature set: {feat_name} ===")

    df_feats = pd.read_csv(feat_path)
    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    # Build stimulus → CIDs for test set
    stimulus_to_cids = {}
    max_cids = 0
    df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]

    for _, row in df_stim_map_filtered.iterrows():
        stimulus = row['id']
        cids = [cid for cid in get_cids(row['components']) if cid in cid_to_feats.index]
        if cids:
            stimulus_to_cids[stimulus] = cids
            max_cids = max(max_cids, len(cids))

    print(f"📌 Max components for {feat_name}: {max_cids}")
    print(f"📌 Fingerprint dim for {feat_name}: {fingerprint_dim}")

    # Build feature matrix (original and reversed)
    X_test_original = []
    X_test_reversed = []
    test_ids = []

    for _, row in df_test.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_cids:
            cids = stimulus_to_cids[stim]
            feat_vecs = [cid_to_feats.loc[cid].values for cid in cids]

            # Original order
            padded_vecs = feat_vecs + [np.zeros(fingerprint_dim)] * (max_cids - len(feat_vecs))
            concat_feat = np.concatenate(padded_vecs)
            X_test_original.append(concat_feat)

            # Reversed order
            reversed_vecs = feat_vecs[::-1] + [np.zeros(fingerprint_dim)] * (max_cids - len(feat_vecs))
            reversed_feat = np.concatenate(reversed_vecs)
            X_test_reversed.append(reversed_feat)

            test_ids.append(stim)
        else:
            print(f"⚠️ No features found for stimulus: {stim}")

    if not stimulus_ids:
        stimulus_ids = test_ids  # Set once for final output

    X_test_original = pd.DataFrame(X_test_original)
    X_test_reversed = pd.DataFrame(X_test_reversed)

    # Load models and predict
    model_subdir = os.path.join(MODEL_DIR, feat_name)
    for model_file in os.listdir(model_subdir):
        if not model_file.endswith(".pkl"):
            continue
        label = model_file.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(model_subdir, model_file)
        print(f"🔍 [{feat_name}] Loading model: {label}")

        model = joblib.load(model_path)
        preds_orig = model.predict(X_test_original)
        preds_rev = model.predict(X_test_reversed)
        preds_avg = (preds_orig + preds_rev) / 2.0

        if label not in final_preds:
            final_preds[label] = preds_avg
        else:
            final_preds[label] += preds_avg  # Sum for averaging across feature sets

# === Average predictions across feature sets ===
for label in final_preds:
    final_preds[label] /= len(FEATURE_SET_PATHS)

# === Save final predictions ===
predictions_df = pd.DataFrame({'stimulus': stimulus_ids})
for label in sorted(final_preds.keys()):
    predictions_df[label] = final_preds[label]

predictions_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final averaged predictions saved to {OUTPUT_CSV}")

