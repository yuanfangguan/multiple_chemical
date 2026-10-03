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
CID_SMILES_PATH = "../../data/raw/CID.csv"
OUTPUT_MODEL_DIR = "models_rata_only"

# === Load data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_rata = pd.read_csv(RATA_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Merge RATA SMILES with CID ===
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner')
df_rata_mapped = df_rata_mapped.drop_duplicates(subset='molecule')

# === Convert RATA float columns ===
rata_feature_columns = [col for col in df_rata_mapped.columns if col not in ('SMILES', 'molecule')]
df_rata_mapped[rata_feature_columns] = df_rata_mapped[rata_feature_columns].apply(pd.to_numeric, errors='coerce')

# === Setup ===
cid_to_feats = df_rata_mapped.set_index('molecule')[rata_feature_columns]
fingerprint_dim = cid_to_feats.shape[1]
augmented_dim = fingerprint_dim + 1  # add dilution

# === Stimulus to component list ===
def get_cid_dilution_pairs(components_str):
    comps = str(components_str).split(';')
    pairs = []
    for comp_id_str in comps:
        if comp_id_str.strip().isdigit():
            comp_id = int(comp_id_str)
            cid = component_to_cid.get(comp_id)
            dilution = component_to_dilution.get(comp_id)
            if cid in cid_to_feats.index and dilution is not None:
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

# === Assemble data ===
usable_rows = []
feature_matrix = []

for _, row in df_train.iterrows():
    stim = row['stimulus']
    if stim in stimulus_to_cid_dilutions:
        pairs = stimulus_to_cid_dilutions[stim]

        def build_vec(pairs):
            vecs = []
            for cid, dilution in pairs:
                f = cid_to_feats.loc[cid].values.astype(np.float32)
                vecs.append(np.append(f, dilution))
            while len(vecs) < max_components:
                vecs.append(np.zeros(augmented_dim, dtype=np.float32))
            return np.concatenate(vecs)

        feature_matrix.append(build_vec(pairs))
        feature_matrix.append(build_vec(pairs[::-1]))
        usable_rows.extend([row, row])

# === Final dataset ===
df_usable = pd.DataFrame(usable_rows)
X = pd.DataFrame(feature_matrix)
label_columns = [col for col in df_train.columns if col != 'stimulus']
y = df_usable[label_columns]

# === Train and Save Models ===
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Save column names
X.columns.to_frame(index=False).to_csv(os.path.join(OUTPUT_MODEL_DIR, "columns.csv"), index=False, header=False)


for label in label_columns:
    print(f"Training model for label: {label}")
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, random_state=42)
    model.fit(X, y[label])
    joblib.dump(model, os.path.join(OUTPUT_MODEL_DIR, f"model_{label}.pkl"))
    print(f"✅ Saved model to {OUTPUT_MODEL_DIR}/model_{label}.pkl")

print("\n✅ RATA-only models trained and saved.")

