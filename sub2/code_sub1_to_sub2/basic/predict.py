import pandas as pd
import xgboost as xgb
import os
import joblib
import numpy as np

# === Paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../../sub2/data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../../sub2/data/raw/TASK2_Component_definition.csv"
FEATURES_DIR = "../../../sub2/data/processed"
FEATURE_FILES = [
    "features_maccs.csv",
    "features_morgan.csv",
    "features_rdkitfp.csv",
    "features_descriptors.csv"
]
MODEL_DIR = "models"
OUTPUT_CSV = "predictions.csv"

# === Load Base Data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))

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

stimulus_to_cids = {
    row['id']: get_cids(row['components'])
    for _, row in df_stim_map.iterrows()
}

stimulus_ids = df_test['stimulus']
predictions = pd.DataFrame({'stimulus': stimulus_ids})

for feature_file in FEATURE_FILES:
    feature_name = feature_file.replace('.csv', '').replace('features_', '')
    print(f"🔄 Processing feature file: {feature_name}")

    df_feats = pd.read_csv(os.path.join(FEATURES_DIR, feature_file))
    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])
    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    feature_matrix = []

    for stim in stimulus_ids:
        cids = stimulus_to_cids.get(stim, [])
        valid_cids = [cid for cid in cids if cid in cid_to_feats.index]

        if valid_cids:
            feat_vecs = [cid_to_feats.loc[cid].values for cid in valid_cids]
            avg_feat = np.max(feat_vecs, axis=0)
            num_cids = len(valid_cids)
            combined_feat = np.append(avg_feat, num_cids)
            feature_matrix.append(combined_feat)
        else:
            print(f"⚠️ Warning: No valid CIDs for stimulus {stim}")
            feature_matrix.append([0] * (fingerprint_dim + 1))

    X_test = pd.DataFrame(feature_matrix, columns=cid_to_feats.columns.tolist() + ['num_cids'])

    print(f"📌 Test feature matrix shape for {feature_name}: {X_test.shape}")

    for root, _, files in os.walk(MODEL_DIR):
        if feature_name in root:
            for model_file in files:
                if model_file.endswith(".pkl") and f"_{feature_name}.pkl" in model_file:
                    model_path = os.path.join(root, model_file)

                    label_name = model_file.replace(f"model_", "").replace(f"_{feature_name}.pkl", "")
                    print(f"Loading model: {model_path} → Label: {label_name}")

                    model = joblib.load(model_path)
                    preds = model.predict(X_test)

                    predictions[f"{label_name}_{feature_name}"] = preds

label_names = sorted(set(col.split("_")[0] for col in predictions.columns if col != 'stimulus'))

print("\n🔄 Assembling final predictions for each label:")
for label in label_names:
    cols_for_label = [col for col in predictions.columns if col.startswith(label + "_")]
    print(f"{label}: using {cols_for_label}")
    predictions[label] = predictions[cols_for_label].mean(axis=1)

ensemble_columns = ['stimulus'] + [label for label in label_names]
predictions[ensemble_columns].to_csv(OUTPUT_CSV, index=False)
print(f"✅ Final assembled predictions saved to {OUTPUT_CSV}")

