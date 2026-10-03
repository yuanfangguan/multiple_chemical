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
    "mordred": "../../data/raw/Mordred_Descriptors.csv",
}

MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Load shared data ===
df_test = pd.read_csv(TEST_PATH)
problematic_stimuli = ['AN873']
df_test = df_test[~df_test['stimulus'].isin(problematic_stimuli)]

df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Helper: get (CID, dilution) pairs for each stimulus ===
def get_cid_dilution_pairs(components_str):
    comps = str(components_str).split(';')
    pairs = []
    for comp_id_str in comps:
        comp_id_str = comp_id_str.strip()
        if comp_id_str.isdigit():
            comp_id = int(comp_id_str)
            cid = component_to_cid.get(comp_id)
            dilution = component_to_dilution.get(comp_id)
            if cid is not None and dilution is not None:
                pairs.append((cid, float(dilution)))
    return pairs

# === Prepare container for predictions ===
stimulus_preds_by_label = {}
stim_num_components = {}  # track number of components per stimulus

# === Process each feature set ===
for feat_name, feat_path in FEATURE_SET_PATHS.items():
    print(f"\n=== Processing feature set: {feat_name} ===")

    # === Load feature data ===
    with open(feat_path, 'r', encoding='utf-8', errors='replace') as f:
        df_feats = pd.read_csv(f)
    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    # === Build mapping from stimulus → (CID, dilution) pairs ===
    df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]
    stimulus_to_pairs = {}
    for _, row in df_stim_map_filtered.iterrows():
        stim = row['id']
        pairs = [(cid, dilution) for cid, dilution in get_cid_dilution_pairs(row['components']) if cid in cid_to_feats.index]
        if pairs:
            stimulus_to_pairs[stim] = pairs
            stim_num_components[stim] = len(pairs)

    # === Predict using "single_chemical" models ===
    model_dir = os.path.join(MODEL_DIR, feat_name, "single_chemical")
    if not os.path.exists(model_dir):
        print(f"⚠️ Missing model directory: {model_dir}")
        continue

    for model_file in os.listdir(model_dir):
        if not model_file.endswith(".pkl"):
            continue
        label = model_file.replace("model_", "").replace(".pkl", "")
        model_path = os.path.join(model_dir, model_file)
        print(f"🔍 [{feat_name}/single_chemical] Loading model: {label}")
        model = joblib.load(model_path)

        for stim in df_test['stimulus']:
            if stim not in stimulus_to_pairs:
                continue
            pairs = stimulus_to_pairs[stim]
            preds = []
            for cid, dilution in pairs:
                feat_vector = cid_to_feats.loc[cid].values.astype(np.float64)
                full_vec = np.concatenate([feat_vector, [dilution, len(pairs)]])
                pred = model.predict(full_vec.reshape(1, -1))[0]
                preds.append(pred)

            if not preds:
                continue

            # Combine per-chemical predictions (avg and max equally weighted)
            avg_pred = np.mean(preds)
            max_pred = np.max(preds)
            final_pred = (avg_pred + max_pred) / 2

            if stim not in stimulus_preds_by_label:
                stimulus_preds_by_label[stim] = {}
            if label not in stimulus_preds_by_label[stim]:
                stimulus_preds_by_label[stim][label] = []
            stimulus_preds_by_label[stim][label].append(final_pred)

# === Combine predictions across all feature sets ===
stimulus_ids = sorted(stimulus_preds_by_label.keys())
label_set = set()
for preds in stimulus_preds_by_label.values():
    label_set.update(preds.keys())
label_list = sorted(label_set)

final_data = {
    'stimulus': stimulus_ids,
    'num_components': [stim_num_components.get(stim, 0) for stim in stimulus_ids]  # record num components
}
for label in label_list:
    final_data[label] = []
    for stim in stimulus_ids:
        preds = stimulus_preds_by_label[stim].get(label, [])
        if preds:
            final_data[label].append(np.mean(preds))
        else:
            final_data[label].append(0.0)

# === Save final predictions ===
predictions_df = pd.DataFrame(final_data)
predictions_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final averaged predictions saved to {OUTPUT_CSV}")
print("🧩 Each row now includes the 'num_components' column for component-based evaluation.")

