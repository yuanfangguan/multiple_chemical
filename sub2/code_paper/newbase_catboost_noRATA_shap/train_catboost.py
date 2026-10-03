import pandas as pd
from catboost import CatBoostRegressor
import os
import joblib
import numpy as np
from joblib import Parallel, delayed
import shap
import matplotlib.pyplot as plt

# === Clean up any existing models ===
os.system("rm -rf model*")

# === Paths ===
TRAIN_PATH = "train.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}
OUTPUT_MODEL_DIR = "models"
SHAP_OUTPUT_DIR = "shap_outputs"

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# === Mappings ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Helper: SHAP analysis ===
def shap_analysis(feature_set_name, variant, label, X, model_dir, num_samples=200):
    model_path = os.path.join(model_dir, f"model_{label}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n=== SHAP Analysis for {feature_set_name} ({variant}) → {label} ===")
    model = joblib.load(model_path)

    X_clean = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    if len(X_clean) > num_samples:
        X_sample = X_clean.sample(num_samples, random_state=42)
    else:
        X_sample = X_clean

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_dir = os.path.join(SHAP_OUTPUT_DIR, feature_set_name, variant)
    os.makedirs(shap_dir, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 6))
    shap_plot_path = os.path.join(shap_dir, f"shap_summary_{label}.png")
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved SHAP summary plot for {label} to {shap_plot_path}")

    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    shap_df["sample_index"] = X_sample.index
    shap_df.to_csv(os.path.join(shap_dir, f"shap_values_{label}.csv"), index=False)
    print(f"✅ Saved SHAP values for {label} to CSV.\n")


# === Generic feature set processor ===
def process_feature_set(name, df_feats):
    print(f"\n=== Processing feature set: {name} ===")

    # Drop SMILES if present
    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    # ✅ Give meaningful column names if numeric
    first_feat_col = df_feats.columns[1]
    if str(first_feat_col).isdigit():
        df_feats.columns = ['molecule'] + [f"{name}_bit{i}" for i in range(df_feats.shape[1]-1)]
        print(f"🧩 Renamed {df_feats.shape[1]-1} fingerprint bits for {name}")
    else:
        print(f"✅ Using descriptor names for {name}")

    cid_to_feats = df_feats.set_index('molecule')
    fingerprint_dim = cid_to_feats.shape[1]
    feature_names = list(cid_to_feats.columns)

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
    for _, row in df_stim_map.iterrows():
        stim = row['id']
        pairs = get_cid_dilution_pairs(row['components'])
        if pairs:
            stimulus_to_cid_dilutions[stim] = pairs

    # === Assemble dataset ===
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

    # ✅ Properly named columns
    avg_columns = [f"{fname}_avg" for fname in feature_names]
    max_columns = [f"{fname}_max" for fname in feature_names]
    extra_columns = ["avg_dilution", "num_chems"]

    X_avg = pd.DataFrame(
        np.hstack([feature_matrix[:, :fingerprint_dim], feature_matrix[:, -2:]]),
        columns=avg_columns + extra_columns
    )
    X_max = pd.DataFrame(
        np.hstack([feature_matrix[:, fingerprint_dim:2*fingerprint_dim], feature_matrix[:, -2:]]),
        columns=max_columns + extra_columns
    )

    # === Model training and SHAP ===
    def train_and_save(label, X, out_dir, variant):
        print(f"Training model for {label} at {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            random_seed=42,
            verbose=100
        )

        X_clean = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        model.fit(X_clean, y[label])
        joblib.dump(model, os.path.join(out_dir, f"model_{label}.pkl"))
        print(f"✅ Saved model for {label} in {out_dir}")

        shap_analysis(name, variant, label, X_clean, out_dir)

    avg_dir = os.path.join(OUTPUT_MODEL_DIR, name, "avg")
    max_dir = os.path.join(OUTPUT_MODEL_DIR, name, "max")

    Parallel(n_jobs=-1)(
        delayed(train_and_save)(label, X_avg, avg_dir, "avg") for label in label_columns
    )
    Parallel(n_jobs=-1)(
        delayed(train_and_save)(label, X_max, max_dir, "max") for label in label_columns
    )


# === Process all feature sets ===
for name, path in FEATURE_SET_PATHS.items():
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        df_feats = pd.read_csv(f)
    process_feature_set(name, df_feats)

print("\n✅ All non-RATA models trained and SHAP analyses completed (avg & max variants, with dilution and num chemicals).")

