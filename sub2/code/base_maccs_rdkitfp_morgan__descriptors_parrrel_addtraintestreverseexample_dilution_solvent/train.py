import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# === Paths ===
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
}
OUTPUT_MODEL_DIR = "models"

# === Fixed solvent list provided by user ===
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
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

# === Prepare mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))
component_to_solvent = dict(zip(df_comp_map['id'], df_comp_map['solvent'].fillna("unknown")))

# === Function to load and train model for each feature set ===
def process_feature_set(name, path):
    print(f"\n=== Processing feature set: {name} ===")
    df_feats = pd.read_csv(path)

    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    # === CID → fingerprint vector ===
    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]
    augmented_dim = fingerprint_dim + 1 + num_solvent_types  # fingerprint + dilution + one-hot solvent

    # === Stimulus to list of (CID, dilution, solvent) ===
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
                if cid is not None and dilution is not None and cid in cid_to_feats.index:
                    pairs.append((cid, float(dilution), solvent))
        return pairs

    stimulus_to_component_info = {}
    max_components = 0

    for _, row in df_stim_map.iterrows():
        stimulus = row['id']
        component_info = get_cid_dilution_solvent_pairs(row['components'])
        if component_info:
            stimulus_to_component_info[stimulus] = component_info
            max_components = max(max_components, len(component_info))

    print(f"📌 Max components per stimulus: {max_components}")
    print(f"📌 Fingerprint dim: {fingerprint_dim}, total per component: {augmented_dim}")
    print(f"📌 Total input dim: {max_components * augmented_dim}")

    # === Build dataset ===
    usable_rows = []
    feature_matrix = []

    for _, row in df_train.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_component_info:
            comp_info_list = stimulus_to_component_info[stim]

            def build_feature_vector(comp_info):
                vecs = []
                for cid, dilution, solvent in comp_info:
                    fingerprint = cid_to_feats.loc[cid].values
                    solvent_vec = one_hot_encode_solvent(solvent)
                    full_vec = np.concatenate([fingerprint, [dilution], solvent_vec])
                    vecs.append(full_vec)
                while len(vecs) < max_components:
                    vecs.append(np.zeros(augmented_dim))
                return np.concatenate(vecs)

            # Original order
            feat_orig = build_feature_vector(comp_info_list)
            feature_matrix.append(feat_orig)
            usable_rows.append(row)

            # Reversed order
            feat_rev = build_feature_vector(comp_info_list[::-1])
            feature_matrix.append(feat_rev)
            usable_rows.append(row)

    df_usable = pd.DataFrame(usable_rows)
    X = pd.DataFrame(feature_matrix)
    label_columns = [col for col in df_train.columns if col != 'stimulus']
    y = df_usable[label_columns]

    # === Train models ===
    model_dir = os.path.join(OUTPUT_MODEL_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    for label in label_columns:
        print(f"Training model for label: {label}")
        y_label = y[label]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
        )
        model.fit(X, y_label)

        model_path = os.path.join(model_dir, f"model_{label}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model to {model_path}")

# === Run for each feature set ===
for name, path in FEATURE_SET_PATHS.items():
    process_feature_set(name, path)

print("\n✅ All models trained and saved for all feature sets.")

