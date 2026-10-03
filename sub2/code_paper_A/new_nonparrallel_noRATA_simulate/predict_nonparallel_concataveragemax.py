import pandas as pd
import os
import joblib
import numpy as np
import itertools

# === File paths ===
TEST_PATH = "test.csv"
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

MODEL_NAME = "combined_no_rata"
MODEL_DIR = f"models/{MODEL_NAME}/softmax_pool"
OUTPUT_CSV = "3d_grid_predictions.csv"
ALPHA = 1.0  # Must match the alpha used during training

# === Load metadata ===
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

# Load CID to SMILES/Label mapping
df_cid_info = pd.read_csv(CID_SMILES_PATH)
cid_to_label = {}
for _, row in df_cid_info.iterrows():
    cid = row['molecule']
    smiles = row['SMILES']
    solvent_info = str(row['is_solvent'])
    
    # Use solvent name if available, otherwise use SMILES
    if "(" in solvent_info and ")" in solvent_info:
        label = solvent_info.split('(')[-1].split(')')[0]
    else:
        label = smiles if pd.notna(smiles) else f"CID:{cid}"
    cid_to_label[cid] = label

# Build component mappings
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Load and Clean Chemical Features ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading feature set: {name}")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except:
        df = pd.read_csv(path, encoding='latin1')
    
    if 'SMILES' in df.columns:
        df = df.drop(columns=['SMILES'])
    
    df.set_index('molecule', inplace=True)
    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient='index')
df_chem_feats = df_chem_feats.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f"✅ Chemical features matrix cleaned: {df_chem_feats.shape}")

# === Build Stimulus → [(CID, base_dilution)] map ===
def get_3_component_pairs(components_str):
    comps = [s.strip() for s in str(components_str).split(';') if s.strip().isdigit()]
    if len(comps) != 3: 
        return None
    
    out = []
    for s in comps:
        comp_id = int(s)
        cid = component_to_cid.get(comp_id)
        dil = component_to_dilution.get(comp_id)
        if cid in df_chem_feats.index and dil is not None:
            out.append((cid, float(dil)))
    
    return out if len(out) == 3 else None

stim2pairs = {}
for _, r in df_stim_map.iterrows():
    pairs = get_3_component_pairs(r['components'])
    if pairs:
        stim2pairs[r['id']] = pairs

print(f"✅ Found {len(stim2pairs)} stimuli with exactly 3 components.")

# === Grid Generation Parameters ===
factors = 2.0 ** np.arange(-5, 6) 

# === Load Models into memory once ===
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

models = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_', '').replace('.pkl', '')
        print(f"Loading model: {label}")
        models[label] = joblib.load(os.path.join(MODEL_DIR, fname))

# === Main Prediction Loop ===
all_results = []

for stim_id, pairs in stim2pairs.items():
    cids = [p[0] for p in pairs]
    base_dils = np.array([p[1] for p in pairs])
    
    # Retrieve labels (SMILES) for the current stimulus
    chem_labels = [cid_to_label.get(cid, f"CID:{cid}") for cid in cids]
    
    print(f"\n🚀 Processing stimulus: {stim_id}")
    print(f"🧪 Components: {', '.join(chem_labels)}")
    
    # Pre-calculate tanh of features for these 3 chemicals
    tanh_vecs = np.tanh(df_chem_feats.loc[cids].values.astype(np.float64)) 

    grid_X = []
    grid_meta = []

    # Cartesian product for the 11x11x11 grid
    for f1, f2, f3 in itertools.product(factors, repeat=3):
        current_dils = base_dils * np.array([f1, f2, f3])
        
        # Softmax weight calculation: weight = exp(alpha * dil) / sum(exp(alpha * dil))
        # This determines which chemical "dominates" the mixture feature vector
        exp_scores = np.exp(ALPHA * current_dils)
        w = exp_scores / (exp_scores.sum() + 1e-9)
        
        # Apply softmax-weighted pooling across the 3 chemicals
        pooled = (w[:, None] * tanh_vecs).sum(axis=0)
        
        # Append avg_dilution and count (3) to complete the input feature vector
        feat = np.concatenate([pooled, [current_dils.mean(), 3]])
        
        grid_X.append(feat)
        # Store metadata including SMILES/labels for the output CSV
        grid_meta.append([
            stim_id, 
            chem_labels[0], chem_labels[1], chem_labels[2],
            f1, f2, f3, 
            current_dils[0], current_dils[1], current_dils[2]
        ])

    X_matrix = np.array(grid_X)
    
    # Construct DataFrame for current stimulus
    meta_cols = [
        'stimulus', 'smiles1', 'smiles2', 'smiles3', 
        'f1', 'f2', 'f3', 'dil1', 'dil2', 'dil3'
    ]
    df_res = pd.DataFrame(grid_meta, columns=meta_cols)
    
    # Perform bulk prediction for all labels (e.g., 'Alcoholic', 'Ammonia', etc.)
    for label, model in models.items():
        df_res[label] = model.predict(X_matrix)
    
    all_results.append(df_res)

# === Save Results ===
if all_results:
    df_final = pd.concat(all_results, ignore_index=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS! Combined grid predictions saved to: {OUTPUT_CSV}")
else:
    print("❌ No data processed. Check your file paths and stimulus definitions.")
