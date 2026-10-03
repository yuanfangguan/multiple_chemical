import pandas as pd
from catboost import CatBoostRegressor
import os
import joblib
import numpy as np
os.system("rm -rf model*")

# === Paths ===
TRAIN_PATH = "train_sub2.csv"
STIMULUS_MAP_PATH = "../../../sub2/data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../../sub2/data/raw/TASK2_Component_definition.csv"
FEATURES_DIR = "../../../sub2/data/processed"
FEATURE_FILES = [
    "features_maccs.csv",
    "features_morgan.csv",
    "features_rdkitfp.csv",
    "features_descriptors.csv"
]
OUTPUT_MODEL_DIR = "models"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# === Load Base Data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

# === Prepare: component ID → CID mapping ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))

# === Prepare: stimulus ID → list of CIDs ===
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

label_columns = [col for col in df_train.columns if col != 'stimulus']

# === Aggregation Methods ===
aggregation_methods = {
    "max": lambda arr: np.max(arr, axis=0),
    "q75": lambda arr: np.percentile(arr, 75, axis=0),
}

# === Main Loop for Feature Files and Aggregations ===
for feature_file in FEATURE_FILES:
    print(f"🔄 Processing feature file: {feature_file}")
    df_feats = pd.read_csv(os.path.join(FEATURES_DIR, feature_file))

    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    cid_to_feats = df_feats.set_index('molecule')

    for agg_name, agg_func in aggregation_methods.items():
        usable_rows = []
        feature_matrix = []

        for _, row in df_train.iterrows():
            stim = row['stimulus']
            cids = stimulus_to_cids.get(stim, [])
            valid_cids = [cid for cid in cids if cid in cid_to_feats.index]

            if valid_cids:
                feat_vecs = [cid_to_feats.loc[cid].values for cid in valid_cids]
                feat_vecs = np.array(feat_vecs)
                agg_feat = agg_func(feat_vecs)
                num_cids = len(valid_cids)
                combined_feat = np.append(agg_feat, num_cids)

                feature_matrix.append(combined_feat)
                usable_rows.append(row)

        if not feature_matrix:
            print(f"⚠️ No usable rows for {feature_file} with {agg_name}. Skipping.")
            continue

        df_usable = pd.DataFrame(usable_rows)
        X = pd.DataFrame(feature_matrix, columns=cid_to_feats.columns.tolist() + ['num_cids'])
        y = df_usable[label_columns]

        print(f"📌 Feature matrix shape for {feature_file} ({agg_name}): {X.shape}")

        # Train and save models
        for label in label_columns:
            print(f"Training model for {label} using {feature_file} ({agg_name})")
            y_label = y[label]

            model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.008,
                depth=6,
                random_seed=42,
                verbose=100  # 输出训练进度
            )

            model.fit(X, y_label)

            model_filename = f"model_{label}_{feature_file.replace('.csv', '')}_{agg_name}.pkl"
            model_path = os.path.join(OUTPUT_MODEL_DIR, model_filename)
            joblib.dump(model, model_path)
            print(f"✅ Saved model to {model_path}")

print("✅ All models trained and saved for all feature sets and aggregation methods.")

