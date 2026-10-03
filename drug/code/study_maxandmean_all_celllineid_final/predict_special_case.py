import pandas as pd
import numpy as np
import os
import sys
import re
from catboost import CatBoostRegressor

# ========================
# --- CONFIGURATION ---
# ========================
validation_file = "../../data/triplet_external_validation.csv"
drug_file = "../../data/drug.csv"
model_dir = "results"
feature_files = [
    "../../data/features/features_maccs.csv",
    "../../data/features/features_morgan.csv",
    "../../data/features/features_rdkitfp.csv",
    "../../data/features/features_descriptors.csv",
    "../../data/features/features_mordred.csv"
]
targets = ["css_ri", "synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
output_file = "all_cell_triplet_predictions.csv"

def normalize_cid(x):
    try: return str(int(float(x)))
    except: return str(x)

# ========================
# --- 1. DATA LOADING ---
# ========================
print("--- Loading Metadata ---")
drug_df = pd.read_csv(drug_file, dtype=str)
drug_map = {str(name).lower(): normalize_cid(cid) for name, cid in zip(drug_df["dname"], drug_df["cid"])}

feature_data_map = {}
for feat_file in feature_files:
    prefix = os.path.splitext(os.path.basename(feat_file))[0]
    print(f"🧩 Loading {prefix}...")
    # Read the file and set molecule as index for fast lookups
    feat_df = pd.read_csv(feat_file, low_memory=False)
    feat_df["molecule"] = feat_df["molecule"].apply(normalize_cid)
    feat_df = feat_df.set_index("molecule")
    
    # CRITICAL: Force all feature columns to numeric, turning strings into NaNs
    feat_df = feat_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    feature_data_map[prefix] = feat_df

# ========================
# --- 2. MODEL LOADING ---
# ========================
print("\n--- Loading Models ---")
loaded_models = {}
model_feature_names = None

for target in targets:
    path = f"{model_dir}/model_{target}.cbm"
    if os.path.exists(path):
        m = CatBoostRegressor().load_model(path)
        loaded_models[target] = m
        if model_feature_names is None:
            model_feature_names = m.feature_names_
    else:
        print(f"⚠️ Warning: Model for {target} not found.")

if not model_feature_names:
    print("❌ Error: No models loaded. Check your results directory.")
    sys.exit()

cell_cols = [c for c in model_feature_names if c.startswith("cell_")]
print(f"✅ Models ready. Expected features: {len(model_feature_names)}")

# ========================
# --- 3. ROBUST FEATURE LOGIC ---
# ========================
def get_triplet_features_dict(cids):
    """
    Safely computes MAX and MEAN features by ensuring numeric types.
    """
    triplet_feats = {}
    for prefix, feat_df in feature_data_map.items():
        # Get existing CIDs or create zero-rows for missing ones
        f_list = []
        for cid in cids:
            if cid in feat_df.index:
                f_list.append(feat_df.loc[cid].values)
            else:
                f_list.append(np.zeros(feat_df.shape[1]))
        
        f_matrix = np.vstack(f_list) # Guaranteed to be (3, num_features)
        
        # Calculate poolings
        f_max = np.max(f_matrix, axis=0)
        f_mean = np.mean(f_matrix, axis=0)
        
        # Map to training names
        for i in range(len(f_max)):
            triplet_feats[f"{prefix}_max_{i}"] = f_max[i]
            triplet_feats[f"{prefix}_mean_{i}"] = f_mean[i]
            
    return triplet_feats

# ========================
# --- 4. PREDICTION LOOP ---
# ========================
print("\n--- Starting Predictions ---")
triplets_to_predict = pd.read_csv(validation_file, names=["d1", "d2", "d3"])
all_results = []

for idx, row in triplets_to_predict.iterrows():
    d_names = [str(row["d1"]).lower(), str(row["d2"]).lower(), str(row["d3"]).lower()]
    cids = [drug_map.get(n) for n in d_names]
    
    if None in cids:
        missing = [d_names[i] for i, c in enumerate(cids) if c is None]
        print(f"❌ Skipping {d_names}: CIDs not found for {missing}")
        continue

    print(f"🧪 Processing Triplet: {' + '.join(d_names)}")
    
    # 1. Map features to dictionary
    X_chem_dict = get_triplet_features_dict(cids)

    # 2. Build Batch DataFrame (forces alignment with training columns)
    batch_df = pd.DataFrame(0.0, index=range(len(cell_cols)), columns=model_feature_names)

    # 3. Fill Chemical Features by NAME (Solves the column-shift/3000 issue)
    for col_name, val in X_chem_dict.items():
        if col_name in batch_df.columns:
            batch_df[col_name] = val

    # 4. Fill Cell Line OHE
    for i, cl_col in enumerate(cell_cols):
        batch_df.loc[i, cl_col] = 1.0

    # 5. Predict
    temp_df = pd.DataFrame({
        "Triplet": " + ".join(d_names),
        "Cell_Line": [c.replace("cell_", "") for c in cell_cols]
    })

    for target, model in loaded_models.items():
        preds = model.predict(batch_df)
        # Cap synergy at logical limits for safety
        temp_df[target] = np.clip(preds, -100, 100)

    all_results.append(temp_df)

# ========================
# --- 5. EXPORT ---
# ========================
if all_results:
    final_df = pd.concat(all_results, axis=0)
    final_df.to_csv(output_file, index=False)
    print(f"\n✨ Predictions saved to {output_file}")
else:
    print("❌ No valid predictions were made.")
