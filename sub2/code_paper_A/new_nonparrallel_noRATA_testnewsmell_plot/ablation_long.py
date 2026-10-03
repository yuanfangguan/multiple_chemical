import pandas as pd
import numpy as np
import joblib
import os
from itertools import combinations
from catboost import Pool  # Required for SHAP/Feature Importance

# === 1. Configuration ===
STIMULUS_ID = "AK432"  # Target mixture to analyze
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
MODEL_DIR = "models/combined_no_rata/softmax_pool"

FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}

# === 2. Load Metadata and Mappings ===
print(f"--- Starting Deep XAI Analysis for {STIMULUS_ID} ---")
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === 3. Load Chemical Features (Robust Encoding) ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading {name}...")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.ParserError):
        print(f"  ⚠️ UTF-8 failed for {name}, trying 'latin1'...")
        df = pd.read_csv(path, encoding='latin1')

    if 'SMILES' in df.columns: 
        df = df.drop(columns=['SMILES'])
    
    df.set_index('molecule', inplace=True)
    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient='index').fillna(0)
fp_dim = df_chem_feats.shape[1]
print(f"✅ Feature Matrix Ready: {df_chem_feats.shape}")

# === 4. Extract Mixture Components ===
def get_pairs(components_str):
    comps = str(components_str).split(';')
    out = []
    for s in comps:
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid in df_chem_feats.index and dil is not None:
                out.append((cid, float(dil)))
    return out

stim_row = df_stim_map[df_stim_map['id'] == STIMULUS_ID]
if stim_row.empty:
    raise ValueError(f"Stimulus ID {STIMULUS_ID} not found in map.")
pairs = get_pairs(stim_row.iloc[0]['components'])
print(f"✅ Found {len(pairs)} components in {STIMULUS_ID}")

# === 5. Generate Subsets and Features ===
def get_all_subsets(iterable):
    s = list(iterable)
    return [list(combo) for r in range(1, len(s) + 1) for combo in combinations(s, r)]

all_subsets = get_all_subsets(pairs)
print(f"🚀 Computing features for {len(all_subsets)} combinations...")

X_list = []
subset_labels = []
alpha = 1.0 

for subset in all_subsets:
    vecs = [df_chem_feats.loc[cid].values.astype(np.float64) for cid, d in subset]
    sc = np.array([d for cid, d in subset])
    
    # Softmax Pooling Logic
    arr = np.vstack(vecs)
    exp_scores = np.exp(alpha * sc)
    w = exp_scores / (exp_scores.sum() + 1e-9)
    pooled = (w[:, None] * np.tanh(arr)).sum(axis=0)
    
    feat = np.concatenate([pooled, [sc.mean(), len(sc)]])
    X_list.append(feat)
    subset_labels.append("+".join([str(cid) for cid, d in subset]))

X_eval = pd.DataFrame(X_list).fillna(0)
feature_names = [f"softmax_fp_{i}" for i in range(fp_dim)] + ["avg_dilution", "num_chems"]
X_eval.columns = feature_names

# === 6. Predictions & Explainable AI (XAI) Analysis ===
report = [f"Ablation & SHAP Feature Contribution Report: {STIMULUS_ID}"]
report.append("="*70)

single_indices = [i for i, s in enumerate(all_subsets) if len(s) == 1]
full_mix_idx = len(all_subsets) - 1

for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_', '').replace('.pkl', '')
        print(f"Processing Model: {label}...")
        
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        preds = model.predict(X_eval)
        
        # Identify Max Individual Driver
        single_preds = preds[single_indices]
        max_idx_local = np.argmax(single_preds)
        max_driver_eval_idx = single_indices[max_idx_local]
        max_cid = all_subsets[max_driver_eval_idx][0][0]
        
        max_val = single_preds[max_idx_local]
        full_val = preds[full_mix_idx]
        avg_single = np.mean(single_preds)
        
        report.append(f"\n[ ODOR: {label.upper()} ]")
        report.append(f"Full Mixture Predicted Score: {full_val:.4f}")
        report.append(f"Max Individual Driver (CID {max_cid}): {max_val:.4f}")

        # Feature Impact: Why did the score change from Single -> Mixture?
        # Use Pool for CatBoost compatibility
        eval_pool = Pool(X_eval.iloc[[max_driver_eval_idx, full_mix_idx]])
        raw_shap = model.get_feature_importance(data=eval_pool, type='ShapValues')
        
        # Difference in contribution (Mixture - Single)
        # Slicing [:, :-1] removes the 'Expected Value' (bias) term
        impact = raw_shap[1, :-1] - raw_shap[0, :-1]
        
        # Sort by absolute impact to find the most significant movers
        top_indices = np.argsort(np.abs(impact))[::-1][:5]
        
        report.append("Top 5 Features Driving Perception Change:")
        for idx in top_indices:
            f_name = feature_names[idx]
            diff = impact[idx]
            dir_icon = "⬆️" if diff > 0 else "⬇️"
            report.append(f"  - {f_name:20}: {dir_icon} impact of {abs(diff):.4f}")

        # Label Findings
        if full_val > (max_val + 0.2):
            report.append(">>> RESULT: EMERGENCE (1+1 > 2) 🚀")
        elif full_val < (avg_single - 0.1) and str(max_cid) in subset_labels[full_mix_idx]:
            report.append(f">>> RESULT: SHIELDING OF CID {max_cid} 🛡️")

# === 7. Export ===
out_file = f"ablation_XAI_{STIMULUS_ID}.txt"
with open(out_file, "w") as f:
    f.write("\n".join(report))

print("-" * 30)
print(f"✅ SUCCESS: Deep report generated as '{out_file}'")
