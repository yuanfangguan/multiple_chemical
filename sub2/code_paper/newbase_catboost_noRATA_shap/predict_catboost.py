import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import os

# ========================
# Paths
# ========================
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"
MODEL_BASE_DIR = "models"
OUTPUT_DIR = "predictions"

FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}

# ========================
# Load shared data
# ========================
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

# Mappings
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================
# Helper to build stimulus → (CID, dilution) map
# ========================
def get_pairs(components_str):
    comps = str(components_str).split(';')
    out = []
    for s in comps:
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid is not None and dil is not None:
                out.append((cid, float(dil)))
    return out


stim2pairs = {}
for _, r in df_stim_map.iterrows():
    pairs = get_pairs(r["components"])
    if pairs:
        stim2pairs[r["id"]] = pairs


# ========================
# Function to predict for one feature set
# ========================
def process_feature_set(feature_set_name, feature_path):
    print(f"\n=== Processing feature set: {feature_set_name} ===")

    # Load feature set
    try:
        df_feats = pd.read_csv(feature_path, encoding="utf-8")
    except UnicodeDecodeError:
        df_feats = pd.read_csv(feature_path, encoding="latin1")

    if "SMILES" in df_feats.columns:
        df_feats = df_feats.drop(columns=["SMILES"])

    # ✅ Ensure consistent feature names
    first_feat_col = df_feats.columns[1]
    if str(first_feat_col).isdigit():
        df_feats.columns = ["molecule"] + [f"{feature_set_name}_bit{i}" for i in range(df_feats.shape[1] - 1)]
        print(f"🧩 Renamed {df_feats.shape[1]-1} fingerprint bits for {feature_set_name}")
    else:
        print(f"✅ Using descriptor names for {feature_set_name}")

    df_feats = df_feats.set_index("molecule")
    fp_dim = df_feats.shape[1]

    # === Build feature vectors for test stimuli ===
    X_list = []
    stim_ids = []
    for _, r in df_test.iterrows():
        stim = r["stimulus"]
        if stim not in stim2pairs:
            continue
        pairs = stim2pairs[stim]
        vecs, dils = [], []
        for cid, d in pairs:
            if cid in df_feats.index:
                vecs.append(df_feats.loc[cid].values.astype(np.float64))
                dils.append(d)
        if vecs:
            stacked = np.vstack(vecs)
            avg_fp = np.mean(stacked, axis=0)
            max_fp = np.max(stacked, axis=0)
            avg_dil = np.mean(dils)
            n_chems = len(dils)
        else:
            avg_fp = np.zeros(fp_dim)
            max_fp = np.zeros(fp_dim)
            avg_dil, n_chems = 0.0, 0

        combined = np.concatenate([avg_fp, max_fp, [avg_dil, n_chems]])
        X_list.append(combined)
        stim_ids.append(stim)

    X_all = np.array(X_list)

    # ✅ Column names same as training
    avg_cols = [f"{c}_avg" for c in df_feats.columns]
    max_cols = [f"{c}_max" for c in df_feats.columns]
    extra_cols = ["avg_dilution", "num_chems"]
    all_cols = avg_cols + max_cols + extra_cols

    X_df = pd.DataFrame(X_all, columns=all_cols)

    # ========================
    # Predict using both avg and max variants
    # ========================
    for variant in ["avg", "max"]:
        model_dir = os.path.join(MODEL_BASE_DIR, feature_set_name, variant)
        if not os.path.isdir(model_dir):
            print(f"⚠️ No model directory found for {feature_set_name}/{variant}")
            continue

        preds_dict = {}
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".pkl"):
                continue
            label = fname.replace("model_", "").replace(".pkl", "")
            model_path = os.path.join(model_dir, fname)
            print(f"📦 Loading model for {label} from {model_path}")
            model = joblib.load(model_path)

            # ✅ Align columns with model
            X_aligned = X_df.reindex(columns=model.feature_names_, fill_value=0)

            preds = model.predict(X_aligned)
            preds_dict[label] = preds

        # === Save results ===
        if preds_dict:
            df_out = pd.DataFrame({"stimulus": stim_ids})
            for label, preds in preds_dict.items():
                df_out[label] = preds
            out_path = os.path.join(OUTPUT_DIR, f"pred_{feature_set_name}_{variant}.tsv")
            df_out.to_csv(out_path, sep="\t", index=False)
            print(f"✅ Saved predictions for {feature_set_name}/{variant} → {out_path}")


# ========================
# Run predictions for all feature sets
# ========================
for feature_set_name, feature_path in FEATURE_SET_PATHS.items():
    process_feature_set(feature_set_name, feature_path)

print("\n✅ All predictions completed and saved.")

