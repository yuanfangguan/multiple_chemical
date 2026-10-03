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

# === Fixed solvent list ===
PREDEFINED_SOLVENTS = [
    "1",
    "90% ethanol",
    "99% ethanol",
    "dep",
    "mineral oil",
    "nt",
    "paraffin oil",
    "pg",
#    "solvent",
    "unknown"
]
solvent_to_onehot = {solvent: i for i, solvent in enumerate(PREDEFINED_SOLVENTS)}
num_solvent_types = len(PREDEFINED_SOLVENTS)

def one_hot_encode_solvent(solvent):
    vec = np.zeros(num_solvent_types)
    index = solvent_to_onehot.get(solvent, solvent_to_onehot["unknown"])
    vec[index] = 1
    return vec

# === Load shared data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))
component_to_solvent = dict(zip(df_comp_map['id'], df_comp_map['solvent'].fillna("unknown")))

# === Helper: Get list of (CID, dilution, solvent)
def get_cid_dilution_solvent_pairs(components_str):
    comps = str(components_str).split(';')
    pairs = []
    for comp_id_str in comps:
        comp_id_str = comp_id_str.strip()
        if comp_id_str.isdigit():
            comp_id = int(comp_id_str)
            cid = component_to_cid.get(comp_id)
            dilution = component_to_dilution.get(comp_id)
            solvent = component_to_solvent.get(comp_id, "unknown")
            if cid is not None and dilution is not None:
                pairs.append((cid, float(dilution), solvent))
    return pairs

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
    augmented_dim = fingerprint_dim + 1 + num_solvent_types  # fingerprint + dilution + solvent

    # Build stimulus → (CID, dilution, solvent) pairs
    stimulus_to_pairs = {}
    max_components = 0
    df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]

    for _, row in df_stim_map_filtered.iterrows():
        stimulus = row['id']
        pairs = [
            (cid, dilution, solvent)
            for cid, dilution, solvent in get_cid_dilution_solvent_pairs(row['components'])
            if cid in cid_to_feats.index
        ]
        if pairs:
            stimulus_to_pairs[stimulus] = pairs
            max_components = max(max_components, len(pairs))

    print(f"📌 Max components for {feat_name}: {max_components}")
    print(f"📌 Augmented dim per component: {augmented_dim}")

    # Build feature matrix (original and reversed)
    X_test_original = []
    X_test_reversed = []
    test_ids = []

    def build_feature_vector(pairs):
        vecs = []
        for cid, dilution, solvent in pairs:
            fingerprint = cid_to_feats.loc[cid].values
            solvent_vec = one_hot_encode_solvent(solvent)
            full_vec = np.concatenate([fingerprint, [dilution], solvent_vec])
            vecs.append(full_vec)
        while len(vecs) < max_components:
            vecs.append(np.zeros(augmented_dim))
        return np.concatenate(vecs)

    for _, row in df_test.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_pairs:
            pairs = stimulus_to_pairs[stim]

            feat_orig = build_feature_vector(pairs)
            feat_rev = build_feature_vector(pairs[::-1])

            X_test_original.append(feat_orig)
            X_test_reversed.append(feat_rev)
            test_ids.append(stim)
        else:
            print(f"⚠️ No features found for stimulus: {stim}")

    if not stimulus_ids:
        stimulus_ids = test_ids

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
            final_preds[label] += preds_avg

# === Average predictions across feature sets ===
for label in final_preds:
    final_preds[label] /= len(FEATURE_SET_PATHS)

# === Save final predictions ===
predictions_df = pd.DataFrame({'stimulus': stimulus_ids})
for label in sorted(final_preds.keys()):
    predictions_df[label] = final_preds[label]

predictions_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final averaged predictions saved to {OUTPUT_CSV}")

