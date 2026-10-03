import pandas as pd
import numpy as np
import joblib
import os
from itertools import combinations

# === 1. Configuration ===
STIMULUS_ID = "AK432"  # The specific mixture to analyze
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

# === 2. Load Mappings ===
print(f"--- Loading Metadata for {STIMULUS_ID} ---")
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === 3. Load Chemical Features with Robust Encoding ===
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
print(f"✅ Feature Matrix Ready: {df_chem_feats.shape}")

# === 4. Extract Components for target Mixture ===
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
    raise ValueError(f"Stimulus ID {STIMULUS_ID} not found!")

pairs = get_pairs(stim_row.iloc[0]['components'])
print(f"✅ Found {len(pairs)} valid chemical components in {STIMULUS_ID}")

# === 5. Generate Power Set (All Subsets) ===
def get_all_subsets(iterable):
    s = list(iterable)
    # Returns all combinations from size 1 up to full size
    return [list(combo) for r in range(1, len(s) + 1) for combo in combinations(s, r)]

all_subsets = get_all_subsets(pairs)
print(f"🚀 Generating predictions for {len(all_subsets)} unique combinations...")

X_list = []
subset_labels = []
alpha = 1.0 # Pooling hyperparameter

for subset in all_subsets:
    vecs = [df_chem_feats.loc[cid].values.astype(np.float64) for cid, d in subset]
    scores = [d for cid, d in subset]
    
    # Softmax Pooling Logic
    arr = np.vstack(vecs)
    sc = np.array(scores)
    exp_scores = np.exp(alpha * sc)
    weights = exp_scores / (exp_scores.sum() + 1e-9)
    
    pooled = (weights[:, None] * np.tanh(arr)).sum(axis=0)
    # Feature vector: [pooled_features, mean_dilution, num_chemicals]
    feat = np.concatenate([pooled, [sc.mean(), len(sc)]])
    
    X_list.append(feat)
    subset_labels.append("+".join([str(cid) for cid, d in subset]))

# Convert to DataFrame and fix Column Names for CatBoost Compatibility
X_eval = pd.DataFrame(X_list).fillna(0)
fp_dim = df_chem_feats.shape[1]
X_eval.columns = [f"softmax_fp_{i}" for i in range(fp_dim)] + ["avg_dilution", "num_chems"]

# === 6. Run Predictions & Generate Deep Report ===
report = [f"--- Comprehensive Ablation Report: {STIMULUS_ID} ---"]
report.append(f"Total Combinations Tested: {len(all_subsets)}\n")

# Indices for single chemicals and the full mixture
single_indices = [i for i, s in enumerate(all_subsets) if len(s) == 1]
full_mix_idx = len(all_subsets) - 1

for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_', '').replace('.pkl', '')
        print(f"Analyzing Odor: {label}...")
        
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        preds = model.predict(X_eval)
        
        # 1. Identify the 'Max Driver' (The single chemical with highest score)
        single_preds = preds[single_indices]
        max_idx_within_singles = np.argmax(single_preds)
        max_single_val = single_preds[max_idx_within_singles]
        # Get the CID of the chemical that generated this max value
        max_cid = all_subsets[single_indices[max_idx_within_singles]][0][0]
        
        avg_single = np.mean(single_preds)
        full_val = preds[full_mix_idx]
        
        report.append(f"\n[ ODOR TYPE: {label.upper()} ]")
        report.append(f"Full Mixture Score: {full_val:.4f}")
        report.append(f"Max Individual Driver: CID {max_cid} (Score: {max_single_val:.4f})")
        report.append(f"Avg Single Component Score: {avg_single:.4f}")

        # --- 2. Identify Emergence (+0.2 over max_single_val) ---
        emergent = []
        for i, val in enumerate(preds):
            if val > (max_single_val + 0.2):
                emergent.append((val, len(all_subsets[i]), subset_labels[i]))
        
        emergent.sort(key=lambda x: x[0], reverse=True)
        if emergent:
            report.append(f"🚀 Found {len(emergent)} emergent subsets (surpassing Max Driver CID {max_cid}). Top 5:")
            for v, c, n in emergent[:5]:
                report.append(f"  Score: {v:.3f} | Size: {c} | Mix: {n}")
        
        # --- 3. Identify Shielding (MUST contain the max_cid) ---
        # Logic: We only care if the combination kills the strongest smell
        shielded = []
        target_str = str(max_cid)
        for i, val in enumerate(preds):
            # Condition: Score is low AND the mixture includes the strongest chemical
            if val < (avg_single - 0.1) and target_str in subset_labels[i]:
                shielded.append((val, len(all_subsets[i]), subset_labels[i]))
        
        shielded.sort(key=lambda x: x[0])
        if shielded:
            report.append(f"🛡️ Found {len(shielded)} subsets that shield/mask CID {max_cid}. Top 5 lowest scores:")
            for v, c, n in shielded[:5]:
                report.append(f"  Score: {v:.3f} | Size: {c} | Mix: {n}")
        else:
            report.append(f"No significant shielding of CID {max_cid} detected.")


# === 7. Export Results ===
output_filename = f"ablation_results_{STIMULUS_ID}.txt"
with open(output_filename, "w") as f:
    f.write("\n".join(report))

print("-" * 30)
print(f"✅ SUCCESS: Full analysis saved to {output_filename}")
