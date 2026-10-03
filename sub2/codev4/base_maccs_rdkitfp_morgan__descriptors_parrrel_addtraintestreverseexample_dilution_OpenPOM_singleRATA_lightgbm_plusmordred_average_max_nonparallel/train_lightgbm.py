import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np
from joblib import Parallel, delayed

# === Paths ===
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
RATA_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
SINGLE_RATA_PATH = "../../data/raw/Task2_single_RATA.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}
OUTPUT_MODEL_DIR = "models"

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_rata_single = pd.read_csv(SINGLE_RATA_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Load and merge all chemical features ===
merged_feats_dict = {}

for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')  # fallback if utf-8 fails

    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])
    df.set_index('molecule', inplace=True)

    for cid in df.index:
        if cid not in merged_feats_dict:
            merged_feats_dict[cid] = []
        merged_feats_dict[cid].extend(df.loc[cid].values.tolist())

merged_cids = list(merged_feats_dict.keys())
merged_feature_vectors = [merged_feats_dict[cid] for cid in merged_cids]
df_chem_feats = pd.DataFrame(merged_feature_vectors, index=merged_cids)
df_chem_feats.index.name = 'molecule'

# === Merge with single RATA ===
df_rata_single_mapped = df_rata_single.merge(df_cid, on='molecule', how='inner').drop_duplicates(subset='molecule')
rata_single_feature_columns = [
    col for col in df_rata_single_mapped.columns 
    if col not in ('SMILES', 'molecule', 'stimulus', 'components', 'dilution')
]
df_rata_single_mapped[rata_single_feature_columns] = df_rata_single_mapped[rata_single_feature_columns].apply(pd.to_numeric, errors='coerce')

df_combined_feats = df_chem_feats.join(
    df_rata_single_mapped.set_index('molecule')[rata_single_feature_columns],
    how='left'
).fillna(0)

# === Define feature processing and model training ===
def process_feature_set(name, df_feats):
    print(f"\n=== Processing combined feature set: {name} ===")

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    # Construct stimulus → (cid, dilution) pairs
    def get_cid_dilution_pairs(components_str):
        comps = str(components_str).split(';')
        pairs = []
        for comp_id_str in comps:
            if comp_id_str.strip().isdigit():
                comp_id = int(comp_id_str)
                cid = component_to_cid.get(comp_id)
                dilution = component_to_dilution.get(comp_id)
                if cid is not None and dilution is not None and cid in cid_to_feats.index:
                    pairs.append((cid, float(dilution)))
        return pairs

    stimulus_to_cid_dilutions = {}
    for _, row in df_stim_map.iterrows():
        stim = row['id']
        pairs = get_cid_dilution_pairs(row['components'])
        if pairs:
            stimulus_to_cid_dilutions[stim] = pairs

    usable_rows = []
    feature_matrix = []

    def build_feature_vector(pairs):
        feature_vecs = []
        dilutions = []
        for cid, dilution in pairs:
            base_feats = cid_to_feats.loc[cid].values.astype(np.float64)
            feature_vecs.append(base_feats)
            dilutions.append(dilution)

        if not feature_vecs:
            avg_fingerprint = np.zeros(fingerprint_dim)
            max_fingerprint = np.zeros(fingerprint_dim)
            avg_dilution = 0.0
            num_chems = 0
        else:
            stacked = np.vstack(feature_vecs)
            avg_fingerprint = np.mean(stacked, axis=0)
            max_fingerprint = np.max(stacked, axis=0)
            avg_dilution = np.mean(dilutions)
            num_chems = len(pairs)

        return np.concatenate([avg_fingerprint, max_fingerprint, [avg_dilution, num_chems]])

    for _, row in df_train.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_cid_dilutions:
            pairs = stimulus_to_cid_dilutions[stim]
            feat_vector = build_feature_vector(pairs)
            feature_matrix.append(feat_vector)
            usable_rows.append(row)

    df_usable = pd.DataFrame(usable_rows)
    label_columns = [col for col in df_train.columns if col != 'stimulus']
    y = df_usable[label_columns]

    feature_matrix = np.array(feature_matrix)
    fingerprint_dim = (feature_matrix.shape[1] - 2) // 2

    X_avg = pd.DataFrame(np.hstack([
        feature_matrix[:, :fingerprint_dim],     # avg
        feature_matrix[:, -2:]                   # dilution, num_chems
    ]))

    X_max = pd.DataFrame(np.hstack([
        feature_matrix[:, fingerprint_dim:2*fingerprint_dim],  # max
        feature_matrix[:, -2:]                                 # dilution, num_chems
    ]))

    def train_and_save(label, X, out_dir):
        print(f"Training model for {label} at {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
        )


        X_clean = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        model.fit(X_clean, y[label])
        joblib.dump(model, os.path.join(out_dir, f"model_{label}.pkl"))
        print(f"✅ Saved model for {label} in {out_dir}")

    avg_dir = os.path.join(OUTPUT_MODEL_DIR, name, "avg")
    max_dir = os.path.join(OUTPUT_MODEL_DIR, name, "max")

    Parallel(n_jobs=-1)(
        delayed(train_and_save)(label, X_avg, avg_dir) for label in label_columns
    )

    Parallel(n_jobs=-1)(
        delayed(train_and_save)(label, X_max, max_dir) for label in label_columns
    )

# === Train using combined features ===
process_feature_set("combined_with_rata", df_combined_feats.reset_index())

print("\n✅ All combined models trained and saved (avg & max with RATA + chemical features).")

