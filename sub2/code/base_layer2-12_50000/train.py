import pandas as pd
import lightgbm as lgb
import os
import joblib
import numpy as np

# === Paths ===
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
FEATURE_DIR = "../../../data_external"  # Base path for feature files

# === Load Data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

# === Prepare: component ID → CID mapping ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))

# === Stimulus to list of CIDs ===
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

stimulus_to_cids = {}
for _, row in df_stim_map.iterrows():
    stimulus = row['id']
    cids = get_cids(row['components'])
    if cids:
        stimulus_to_cids[stimulus] = cids

label_columns = [col for col in df_train.columns if col != 'stimulus']

# === Iterate over layers 2 to 12 ===
for layer in range(2, 13):
    print(f"\n🔍 Training on features from layer {layer}")
    FEATURES_PATH = os.path.join(FEATURE_DIR, f"50000_sub2_cid_smiles_hidden_features_layer{layer}.csv")
    OUTPUT_MODEL_DIR = f"models_layer{layer}"
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    df_feats = pd.read_csv(FEATURES_PATH)
    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]

    # Calculate max_cids for this layer
    max_cids = max(len(cids) for cids in stimulus_to_cids.values())

    print(f"📌 Max components per stimulus: {max_cids}")
    print(f"📌 Fingerprint dimension per CID: {fingerprint_dim}")
    print(f"📌 Total input dimension: {max_cids * fingerprint_dim}")

    # === Build final dataset for this layer ===
    usable_rows = []
    feature_matrix = []

    for _, row in df_train.iterrows():
        stim = row['stimulus']
        if stim in stimulus_to_cids:
            cids = [cid for cid in stimulus_to_cids[stim] if cid in cid_to_feats.index]
            if not cids:
                continue

            feat_vecs = [cid_to_feats.loc[cid].values for cid in cids]

            while len(feat_vecs) < max_cids:
                feat_vecs.append(np.zeros(fingerprint_dim))

            concat_feat = np.concatenate(feat_vecs)
            feature_matrix.append(concat_feat)
            usable_rows.append(row)

    df_usable = pd.DataFrame(usable_rows)
    X = pd.DataFrame(feature_matrix)
    y = df_usable[label_columns]

    # === Train a model per label ===
    for label in label_columns:
        print(f"Training model for: {label}")
        y_label = y[label]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            random_state=42
        )

        model.fit(X, y_label)

        model_path = os.path.join(OUTPUT_MODEL_DIR, f"model_{label}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model to {model_path}")

print("\n✅ All layers trained and models saved.")

