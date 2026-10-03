import pandas as pd
import numpy as np
import os
import joblib

# === Paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
RATA_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
MODEL_DIR = "models_rata_only"
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_rata = pd.read_csv(RATA_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Load column names from training
columns_path = os.path.join(MODEL_DIR, "columns.csv")
feature_columns = pd.read_csv(columns_path, header=None).iloc[:, 0].tolist()


# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Merge RATA with CID
df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner').drop_duplicates(subset='molecule')
rata_feature_columns = [col for col in df_rata_mapped.columns if col not in ('SMILES', 'molecule')]
df_rata_mapped[rata_feature_columns] = df_rata_mapped[rata_feature_columns].apply(pd.to_numeric, errors='coerce')

# === Setup
cid_to_feats = df_rata_mapped.set_index('molecule')[rata_feature_columns]
fingerprint_dim = cid_to_feats.shape[1]
augmented_dim = fingerprint_dim + 1

# === Stimulus to component list
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

stimulus_to_pairs = {}
max_components = 0
for _, row in df_stim_map.iterrows():
    stim = row['id']
    pairs = get_cid_dilution_pairs(row['components'])
    if pairs:
        stimulus_to_pairs[stim] = pairs
        max_components = max(max_components, len(pairs))

# === Build test feature matrix
X_test = []
stimulus_ids = []
for _, row in df_test.iterrows():
    stim = row['stimulus']
    if stim in stimulus_to_pairs:
        pairs = stimulus_to_pairs[stim]

        def build_vec(pairs):
            vecs = []
            for cid, dilution in pairs:
                f = cid_to_feats.loc[cid].values.astype(np.float32)
                vecs.append(np.append(f, dilution))
            while len(vecs) < max_components:
                vecs.append(np.zeros(augmented_dim, dtype=np.float32))
            return np.concatenate(vecs)

        fwd = build_vec(pairs)
        rev = build_vec(pairs[::-1])
        X_test.append((stim, fwd, rev))
    else:
        print(f"⚠️ Skipped unknown stimulus: {stim}")

# === Predict
predictions = {}
for model_file in os.listdir(MODEL_DIR):
    if not model_file.endswith(".pkl"):
        continue
    label = model_file.replace("model_", "").replace(".pkl", "")
    model_path = os.path.join(MODEL_DIR, model_file)
    print(f"🔍 Predicting label: {label}")
    model = joblib.load(model_path)

    preds = []
    for stim, fwd, rev in X_test:
        p1 = model.predict(pd.DataFrame([fwd], columns=feature_columns))[0]
        p2 = model.predict(pd.DataFrame([rev], columns=feature_columns))[0]
        avg = 0.5 * (p1 + p2)
        preds.append((stim, avg))

    predictions[label] = dict(preds)

# === Assemble final DataFrame
stimuli = [stim for stim, _, _ in X_test]
final_df = pd.DataFrame({'stimulus': stimuli})
for label in sorted(predictions.keys()):
    final_df[label] = final_df['stimulus'].map(predictions[label])

# === Save
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Predictions saved to {OUTPUT_CSV}")

