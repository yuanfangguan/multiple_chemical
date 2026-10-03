import pandas as pd
import os
import joblib
import numpy as np

# === File paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
RATA_PATH = "../../data/raw/OpenPOM_Dream_RATA.csv"
SINGLE_RATA_PATH = "../../data/raw/Task2_single_RATA.csv"

FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv",
    "rata": RATA_PATH,
    "rata_single": SINGLE_RATA_PATH,
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
df_rata = pd.read_csv(RATA_PATH)
df_rata_single = pd.read_csv(SINGLE_RATA_PATH)

component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

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

# === Prepare final predictions container ===
stimulus_ids = []
final_preds = {}

# === For each feature set ===
for feat_name, feat_path in FEATURE_SET_PATHS.items():
    print(f"\n=== Processing feature set: {feat_name} ===")

    # === Load features
    if feat_name == "rata":
        df_rata_mapped = df_rata.merge(df_cid, on='SMILES', how='inner').drop_duplicates(subset='molecule')
        rata_feature_columns = [col for col in df_rata_mapped.columns if col not in ('SMILES', 'molecule')]
        df_rata_mapped[rata_feature_columns] = df_rata_mapped[rata_feature_columns].apply(pd.to_numeric, errors='coerce')
        df_feats = df_rata_mapped[['molecule'] + rata_feature_columns]
    elif feat_name == "rata_single":
        df_rata_single_mapped = df_rata_single.merge(df_cid, on='molecule', how='inner').drop_duplicates(subset='molecule')
        rata_single_feature_columns = [col for col in df_rata_single_mapped.columns if col not in ('SMILES', 'molecule', "stimulus", "components", "dilution")]
        df_rata_single_mapped[rata_single_feature_columns] = df_rata_single_mapped[rata_single_feature_columns].apply(pd.to_numeric, errors='coerce')
        df_feats = df_rata_single_mapped[['molecule'] + rata_single_feature_columns]
    else:
        with open(feat_path, 'r', encoding='utf-8', errors='replace') as f:
            df_feats = pd.read_csv(f)
        if 'SMILES' in df_feats.columns:
            df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    # === Build stimulus → (CID, dilution) pairs
    stimulus_to_pairs = {}
    df_stim_map_filtered = df_stim_map[df_stim_map['id'].isin(df_test['stimulus'])]

    for _, row in df_stim_map_filtered.iterrows():
        stim = row['id']
        pairs = [(cid, dilution) for cid, dilution in get_cid_dilution_pairs(row['components']) if cid in cid_to_feats.index]
        if pairs:
            stimulus_to_pairs[stim] = pairs

    # === Build test feature matrices (avg and max)
    X_test_avg = []
    X_test_max = []
    test_ids = []

    def build_feature_vectors(pairs):
        vecs = []
        dilutions = []
        for cid, dilution in pairs:
            f = cid_to_feats.loc[cid].values.astype(np.float64)
            vecs.append(f)
            dilutions.append(dilution)
        if not vecs:
            avg = np.zeros(fingerprint_dim)
            maxv = np.zeros(fingerprint_dim)
            avg_dilution = 0.0
            num_chems = 0
        else:
            stacked = np.vstack(vecs)
            avg = np.mean(stacked, axis=0)
            maxv = np.max(stacked, axis=0)
            avg_dilution = np.mean(dilutions)
            num_chems = len(pairs)
        return np.concatenate([maxv, [avg_dilution, num_chems]]), np.concatenate([maxv, [avg_dilution, num_chems]])

    for _, row in df_test.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_pairs:
            avg_vec, max_vec = build_feature_vectors(stimulus_to_pairs[stim])
            X_test_avg.append(max_vec)
            X_test_max.append(max_vec)
            test_ids.append(stim)
        else:
            print(f"⚠️ No features found for stimulus: {stim}")

    if not stimulus_ids:
        stimulus_ids = test_ids

    X_test_avg = pd.DataFrame(X_test_avg).apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_max = pd.DataFrame(X_test_max).apply(pd.to_numeric, errors='coerce').fillna(0)

    # === Load and predict for both avg and max models
    for version, X in [("avg", X_test_avg), ("max", X_test_max)]:
        model_dir = os.path.join(MODEL_DIR, feat_name, version)
        if not os.path.exists(model_dir):
            print(f"⚠️ Missing model directory: {model_dir}")
            continue
        for model_file in os.listdir(model_dir):
            if not model_file.endswith(".pkl"):
                continue
            label = model_file.replace("model_", "").replace(".pkl", "")
            model_path = os.path.join(model_dir, model_file)
            print(f"🔍 [{feat_name}/{version}] Loading model: {label}")
            model = joblib.load(model_path)
            preds = model.predict(X)

            key = f"{label}"
            if key not in final_preds:
                final_preds[key] = preds
            else:
                final_preds[key] += preds

# === Average predictions across feature sets × 2 (avg + max)
num_model_variants = len(FEATURE_SET_PATHS) * 2
for label in final_preds:
    final_preds[label] /= num_model_variants

# === Save final predictions
predictions_df = pd.DataFrame({'stimulus': stimulus_ids})
for label in sorted(final_preds.keys()):
    predictions_df[label] = final_preds[label]

predictions_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final averaged predictions saved to {OUTPUT_CSV}")

