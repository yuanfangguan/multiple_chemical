import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

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
}
OUTPUT_MODEL_DIR = "models"

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_rata = pd.read_csv(RATA_PATH)
df_rata_single = pd.read_csv(SINGLE_RATA_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Generic feature set processor ===
def process_feature_set(name, df_feats):
    print(f"\n=== Processing feature set: {name} ===")

    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]
    augmented_dim = fingerprint_dim + 1

    # === Construct stimulus → (cid, dilution) pairs ===
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
    max_components = 0
    for _, row in df_stim_map.iterrows():
        stim = row['id']
        pairs = get_cid_dilution_pairs(row['components'])
        if pairs:
            stimulus_to_cid_dilutions[stim] = pairs
            max_components = max(max_components, len(pairs))

    # === Assemble dataset ===
    usable_rows = []
    feature_matrix = []

    def build_feature_vector(pairs):
        vecs = []
        for cid, dilution in pairs:
            f = cid_to_feats.loc[cid].values
            vecs.append(np.append(f, dilution))
        while len(vecs) < max_components:
            vecs.append(np.zeros(augmented_dim))
        return np.concatenate(vecs)

    for _, row in df_train.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_cid_dilutions:
            pairs = stimulus_to_cid_dilutions[stim]
            feature_matrix.append(build_feature_vector(pairs))
            feature_matrix.append(build_feature_vector(pairs[::-1]))
            usable_rows.extend([row, row])

    # === Train and save ===
    df_usable = pd.DataFrame(usable_rows)
    X = pd.DataFrame(feature_matrix)
    label_columns = [col for col in df_train.columns if col != 'stimulus']
    y = df_usable[label_columns]

    model_dir = os.path.join(OUTPUT_MODEL_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    X.columns.to_frame(index=False).to_csv(os.path.join(model_dir, "columns.csv"), index=False, header=False)

    for label in label_columns:
        print(f"Training model for label: {label}")
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, random_state=42)
        model.fit(X, y[label])
        joblib.dump(model, os.path.join(model_dir, f"model_{label}.pkl"))
        print(f"✅ Saved model to {model_dir}/model_{label}.pkl")

# === Process standard feature sets ===
for name, path in FEATURE_SET_PATHS.items():
    df_feats = pd.read_csv(path)
    process_feature_set(name, df_feats)

# === Special handling for RATA ===
print("\n=== Processing RATA feature set ===")
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner').drop_duplicates(subset='molecule')
rata_feature_columns = [col for col in df_rata_mapped.columns if col not in ('SMILES', 'molecule')]
df_rata_mapped[rata_feature_columns] = df_rata_mapped[rata_feature_columns].apply(pd.to_numeric, errors='coerce')
df_feats_rata = df_rata_mapped[['molecule'] + rata_feature_columns]
process_feature_set("rata", df_feats_rata)

# === Special handling for single RATA ===
print("\n=== Processing RATA single feature set ===")
df_rata_single_mapped = df_rata_single.merge(df_cid, on='molecule', how='inner').drop_duplicates(subset='molecule')
rata_single_feature_columns = [col for col in df_rata_single_mapped.columns if col not in ('SMILES', 'molecule',"stimulus","components","dilution")]
df_rata_single_mapped[rata_single_feature_columns] = df_rata_single_mapped[rata_single_feature_columns].apply(pd.to_numeric, errors='coerce')
df_feats_single_rata = df_rata_single_mapped[['molecule'] + rata_single_feature_columns]
process_feature_set("rata_single", df_feats_single_rata)


print("\n✅ All models trained and saved (including RATA).")

