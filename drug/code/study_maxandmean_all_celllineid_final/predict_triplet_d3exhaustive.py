import pandas as pd
import numpy as np
import os
import sys
import joblib
import re
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder

# ========================
# --- CONFIGURATION ---
# ========================
train_file = "train.csv"
drug_file = "../../data/drug.csv"
model_dir = "results"
output_dir = "results/three_drug_predictions_d3ext"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

feature_files = [
    "../../data/features/features_maccs.csv",
    "../../data/features/features_morgan.csv",
    "../../data/features/features_rdkitfp.csv",
    "../../data/features/features_descriptors.csv",
    "../../data/features/features_mordred.csv"
]
targets = ["css_ri", "synergy_zip", "synergy_loewe", "synergy_hsa", "synergy_bliss"]
EXCLUDED_CELL_LINES = {"DIPG25"}

# ========================
# --- HELPER FUNCTIONS ---
# ========================

def normalize_cid(x):
    try: return str(int(float(x)))
    except: return str(x)

def merge_one_feature(df, feat_df_numeric, feat_file):
    """
    Helper for Training: Merges features and creates MAX/MEAN columns.
    """
    feat_prefix = os.path.splitext(os.path.basename(feat_file))[0]
    
    # Merge row and col drugs
    df1 = df.merge(feat_df_numeric, left_on="cid_row", right_index=True, how="left")
    df2 = df1.merge(feat_df_numeric, left_on="cid_col", right_index=True, how="left", suffixes=("_row", "_col"))

    base_feat_names = feat_df_numeric.columns
    
    # Create aligned names
    max_names = [f"{feat_prefix}_max_{i}" for i in range(len(base_feat_names))]
    mean_names = [f"{feat_prefix}_mean_{i}" for i in range(len(base_feat_names))]
    
    X_max = pd.DataFrame(np.maximum(df2[[f"{c}_row" for c in base_feat_names]].values, 
                                    df2[[f"{c}_col" for c in base_feat_names]].values), columns=max_names)
    X_mean = pd.DataFrame((df2[[f"{c}_row" for c in base_feat_names]].values + 
                           df2[[f"{c}_col" for c in base_feat_names]].values) / 2, columns=mean_names)

    return pd.concat([X_max, X_mean], axis=1)

# ====================================================
# --- PART 1: DATA PREPARATION & TRAINING ---
# ====================================================

try:
    shard_id = int(sys.argv[1])
    total_shards = int(sys.argv[2])
except:
    shard_id = 0
    total_shards = 1

print(f"--- LOADING DATA (Shard {shard_id+1}/{total_shards}) ---")
train_df = pd.read_csv(train_file, low_memory=False)
drug_df = pd.read_csv(drug_file, dtype=str)
drug_map = dict(zip(drug_df["dname"], drug_df["cid"].apply(normalize_cid)))

train_df["cid_row"] = train_df["drug_row"].map(drug_map)
train_df["cid_col"] = train_df["drug_col"].map(drug_map)
train_df = train_df.dropna(subset=["cid_row", "cid_col"])

feature_arrays = {} 
all_train_feats = []
valid_cids = None

for feat_file in feature_files:
    print(f"🧩 Processing {feat_file} ...")
    feat_df = pd.read_csv(feat_file, low_memory=False)
    feat_df["molecule"] = feat_df["molecule"].apply(normalize_cid)
    prefix = os.path.splitext(os.path.basename(feat_file))[0]
    
    # CRITICAL: Force numeric and fill NaNs to prevent str/int error
    feat_numeric = feat_df.set_index("molecule").apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    # Store for Part 2
    feature_arrays[prefix] = {
        'values': feat_numeric.values, 
        'idx_map': {cid: i for i, cid in enumerate(feat_numeric.index)},
        'num_feats': feat_numeric.shape[1]
    }

    # Intersect for valid drugs list
    cids_set = set(feat_numeric.index)
    valid_cids = cids_set if valid_cids is None else valid_cids.intersection(cids_set)

    # Build Training Matrix
    X_train_part = merge_one_feature(train_df, feat_numeric, feat_file)
    all_train_feats.append(X_train_part)

X_train = pd.concat(all_train_feats, axis=1)

print("🧬 Encoding cell lines...")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(train_df[["cell_line_name"]])
train_ohe = pd.DataFrame(ohe.transform(train_df[["cell_line_name"]]), 
                         columns=[f"cell_{c}" for c in ohe.categories_[0]])
X_train = pd.concat([X_train, train_ohe.reset_index(drop=True)], axis=1)

# Save the exact column order to prevent mismatch in Part 2
final_feature_names = X_train.columns.tolist()
joblib.dump(final_feature_names, f"{model_dir}/feature_names.pkl")

# Train Models (if shard 0)
for target in targets:
    m_path = f"{model_dir}/model_{target}.cbm"
    if shard_id == 0 and not os.path.exists(m_path):
        train_df[target] = pd.to_numeric(train_df[target], errors='coerce')
        mask = train_df[target].notna()
        print(f"🚀 Training {target} model...")
        model = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, verbose=0)
        model.fit(X_train.loc[mask], train_df.loc[mask, target].values)
        model.save_model(m_path)

# =======================================================
# --- PART 2: PARALLELIZED EXHAUSTIVE EXPANSION ---
# =======================================================

def get_pooled_features_batch_safe(d1_cid, d2_cid, d3_list, total_cols):
    """
    Constructs a batch matrix ensuring chemical features align with training.
    """
    n_samples = len(d3_list)
    X_batch = np.zeros((n_samples, total_cols))
    
    curr_col = 0
    for prefix, data in feature_arrays.items():
        vals, idx_map, n_f = data['values'], data['idx_map'], data['num_feats']
        
        f1 = vals[idx_map[d1_cid]] if d1_cid in idx_map else np.zeros(n_f)
        f2 = vals[idx_map[d2_cid]] if d2_cid in idx_map else np.zeros(n_f)
        pair_max = np.maximum(f1, f2)
        pair_sum = f1 + f2
        
        # Batch extract f3
        f3_matrix = np.zeros((n_samples, n_f))
        for i, cid in enumerate(d3_list):
            if cid in idx_map:
                f3_matrix[i] = vals[idx_map[cid]]
        
        # Compute Triplet Pooling
        batch_max = np.maximum(pair_max, f3_matrix)
        batch_mean = (pair_sum + f3_matrix) / 3.0
        
        # Fill strictly into the reserved slots for this feature block
        X_batch[:, curr_col : curr_col + n_f] = batch_max
        X_batch[:, curr_col + n_f : curr_col + 2*n_f] = batch_mean
        curr_col += 2*n_f
        
    return X_batch

global_available_drugs = sorted(list(valid_cids))
cell_lines = sorted(train_df["cell_line_name"].dropna().unique())
my_cells = [c for i, c in enumerate(cell_lines) if i % total_shards == shard_id and c not in EXCLUDED_CELL_LINES]
cell_ohe_map = {cl: ohe.transform([[cl]])[0] for cl in cell_lines}

for target in targets:
    model_path = f"{model_dir}/model_{target}.cbm"
    if not os.path.exists(model_path): continue
    
    model = CatBoostRegressor().load_model(model_path)
    seeds = train_df.dropna(subset=[target])
    shard_results = []
    output_path = f"{output_dir}/exhaustive_{target}_shard_{shard_id}.csv"

    for cl in my_cells:
        cl_seeds = seeds[seeds["cell_line_name"] == cl]
        if cl_seeds.empty: continue
        cl_vector = cell_ohe_map[cl]
        
        for _, row in cl_seeds.iterrows():
            d1, d2 = row["cid_row"], row["cid_col"]
            d3_candidates = [d for d in global_available_drugs if d not in [d1, d2]]
            
            # Generate feature matrix
            X_triplets = get_pooled_features_batch_safe(d1, d2, d3_candidates, len(final_feature_names))
            
            # Identify OHE start position (tail of the feature list)
            ohe_start = len(final_feature_names) - len(cl_vector)
            X_triplets[:, ohe_start:] = cl_vector
            
            # Batch Prediction
            preds = model.predict(X_triplets)
            
            # Record best D3
            best_idx = np.argmax(preds)
            shard_results.append({
                "cell_line": cl, "drug1": d1, "drug2": d2, 
                "pair_score": row[target], "added_drug": d3_candidates[best_idx], 
                "triplet_score": preds[best_idx], "diff": preds[best_idx] - row[target]
            })
            
        pd.DataFrame(shard_results).to_csv(output_path, index=False)
        print(f"Shard {shard_id}: ✅ {cl} processed.")

print(f"✨ Shard {shard_id} Finished.")
