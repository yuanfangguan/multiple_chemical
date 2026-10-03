import pandas as pd
import numpy as np
import os
import sys
import random
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder

# ========================
# --- CONFIGURATION ---
# ========================
train_file = "train.csv"
drug_file = "../../data/drug.csv"
model_dir = "results"
output_dir = "results/random_background"
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
NUM_RANDOM_SAMPLES = 1000 

# ========================
# --- HELPER FUNCTIONS ---
# ========================

def normalize_cid(x):
    try:
        return str(int(float(x)))
    except:
        return str(x)

def get_pooled_features_batch(d1_list, d2_list, d3_list):
    """
    Generalized Batch Pooling for Triplets.
    """
    all_max = []
    all_mean = []
    
    for prefix, feat_data in feature_arrays.items():
        vals = feat_data['values']
        idx_map = feat_data['idx_map']
        
        def get_f(cid): return vals[idx_map[cid]] if cid in idx_map else np.zeros(vals.shape[1])
        
        f1 = np.array([get_f(c) for c in d1_list])
        f2 = np.array([get_f(c) for c in d2_list])
        f3 = np.array([get_f(c) for c in d3_list])
        
        triplet_max = np.maximum(np.maximum(f1, f2), f3)
        triplet_mean = (f1 + f2 + f3) / 3.0
        
        all_max.append(triplet_max)
        all_mean.append(triplet_mean)

    return np.hstack(all_max + all_mean)

# ====================================================
# --- PART 1: DATA & MODEL LOADING ---
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

# Feature Loading and Column Name Reconstruction
feature_arrays = {} 
valid_cids = None
all_chem_feature_names = []

for feat_file in feature_files:
    print(f"🧩 Loading {feat_file} ...")
    feat_df = pd.read_csv(feat_file, low_memory=False)
    feat_df["molecule"] = feat_df["molecule"].apply(normalize_cid)
    prefix = os.path.splitext(os.path.basename(feat_file))[0]
    
    # Store names for alignment
    base_cols = [c for c in feat_df.columns if c != "molecule"]
    all_chem_feature_names.extend([f"{prefix}_max_{i}" for i in range(len(base_cols))])
    all_chem_feature_names.extend([f"{prefix}_mean_{i}" for i in range(len(base_cols))])

    vals = feat_df.drop(columns=["molecule"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    idx_map = {cid: i for i, cid in enumerate(feat_df["molecule"])}
    feature_arrays[prefix] = {'values': vals, 'idx_map': idx_map}

    cids = set(feat_df["molecule"])
    valid_cids = cids if valid_cids is None else valid_cids.intersection(cids)

# Cell Line Encoding
print("🧬 Encoding cell lines...")
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(train_df[["cell_line_name"]])
cell_lines = sorted(train_df["cell_line_name"].dropna().unique())
cell_ohe_map = {cl: ohe.transform([[cl]])[0] for cl in cell_lines}
cell_feature_names = [f"cell_{c}" for c in ohe.categories_[0]]

# Load Pre-trained Models
models = {}
model_feature_names = None
for target in targets:
    path = f"{model_dir}/model_{target}.cbm"
    if os.path.exists(path):
        m = CatBoostRegressor().load_model(path)
        models[target] = m
        if model_feature_names is None:
            model_feature_names = m.feature_names_
    else:
        print(f"⚠️ Warning: Model for {target} not found.")

# =======================================================
# --- PART 2: RANDOM BACKGROUND GENERATION ---
# =======================================================

global_available_drugs = sorted(list(valid_cids))
my_cells = [c for i, c in enumerate(cell_lines) if i % total_shards == shard_id]

print(f"🎲 Generating {NUM_RANDOM_SAMPLES} random combinations...")
random.seed(42 + shard_id)

random_data = []
for _ in range(NUM_RANDOM_SAMPLES):
    cl = random.choice(my_cells)
    d_triplet = random.sample(global_available_drugs, 3)
    random_data.append({
        'cell_line': cl,
        'd1': d_triplet[0],
        'd2': d_triplet[1],
        'd3': d_triplet[2]
    })

random_df = pd.DataFrame(random_data)

# 1. Batch compute chemical features
X_chem_raw = get_pooled_features_batch(random_df['d1'], random_df['d2'], random_df['d3'])
X_chem_df = pd.DataFrame(X_chem_raw, columns=all_chem_feature_names)

# 2. Prepare OHE Cell Line vectors as DataFrame
cl_vectors_df = pd.DataFrame([cell_ohe_map[c] for c in random_df['cell_line']], 
                             columns=cell_feature_names)

# 3. Concatenate and ALIGN with Model
X_full_raw = pd.concat([X_chem_df, cl_vectors_df], axis=1)

print(f"📏 Aligning features: Raw {X_full_raw.shape[1]} -> Model {len(model_feature_names)}")
X_predict = X_full_raw.reindex(columns=model_feature_names, fill_value=0)

# 4. Batch Predict
for target, model in models.items():
    print(f"   -> Predicting {target}...")
    random_df[target] = model.predict(X_predict)

# Save
out_path = f"{output_dir}/random_distribution_shard_{shard_id}.csv"
random_df.to_csv(out_path, index=False)

print(f"✨ Shard {shard_id} Finished. Results saved to {out_path}")
