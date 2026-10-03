import pandas as pd
import numpy as np

# === Paths ===
TEST_PATH = "test.csv" # The original test file with ground truth
MIX_FILE = "predictions_mixtures.csv"
SINGLES_FILE = "predictions_single_chemicals.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"

# === Criteria Settings ===
MARGIN = 0.2  
OUTPUT_EMERGENT_CSV = "synergistic_smells_detailed.csv"

# === 1. Load Data & Rebuild Mappings ===
df_test_gold = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)

component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

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

stim2pairs = {r['id']: get_pairs(r['components']) for _, r in df_stim_map.iterrows() if get_pairs(r['components'])}

# === 2. Load Prediction Data ===
df_mix = pd.read_csv(MIX_FILE)
df_singles = pd.read_csv(SINGLES_FILE)
odor_cols = [c for c in df_mix.columns if c != 'stimulus']

# Build lookup key: "CID_Dilution"
df_singles['lookup_key'] = df_singles['molecule'].astype(str) + "_" + df_singles['dilution'].astype(str)
singles_lookup = df_singles.set_index('lookup_key')[odor_cols]

# Set index for gold standard for easy lookup: stimulus -> odor values
df_gold_lookup = df_test_gold.set_index('stimulus')

# === 3. Identify Synergistic Smells with Gold Standard ===
results = []

for _, mix_row in df_mix.iterrows():
    stim_id = mix_row['stimulus']
    if stim_id not in stim2pairs: continue
    
    comp_keys = [f"{cid}_{dil}" for cid, dil in stim2pairs[stim_id]]
    available_keys = singles_lookup.index.intersection(comp_keys)
    
    if available_keys.empty: continue
    comp_preds = singles_lookup.loc[available_keys]
    
    for odor in odor_cols:
        mix_val = mix_row[odor]
        max_comp_val = comp_preds[odor].max()
        
        if mix_val > (max_comp_val + MARGIN):
            # Fetch the Gold Standard value for this stimulus and odor
            gold_val = np.nan
            if stim_id in df_gold_lookup.index and odor in df_gold_lookup.columns:
                gold_val = df_gold_lookup.loc[stim_id, odor]

            details = {key: round(comp_preds.loc[key, odor], 4) for key in available_keys}
            
            results.append({
                'stimulus': stim_id,
                'odor_type': odor,
                'gold_standard': gold_val,  # <--- New Column
                'mixture_pred': round(mix_val, 4),
                'max_component_pred': round(max_comp_val, 4),
                'improvement': round(mix_val - max_comp_val, 4),
                'num_components': len(available_keys),
                'component_scores': str(details)
            })

# === 4. Save and Sort ===
df_final = pd.DataFrame(results)

if not df_final.empty:
    # Sort by improvement
    df_final = df_final.sort_values(by='improvement', ascending=False)
    df_final.to_csv(OUTPUT_EMERGENT_CSV, index=False)
    
    print(f"✅ Found {len(df_final)} synergistic smell instances.")
    print(f"Results with Gold Standard saved to {OUTPUT_EMERGENT_CSV}")
    
    # Display preview
    print("\nPreview (Gold Standard vs Prediction):")
    cols_to_show = ['stimulus', 'odor_type', 'gold_standard', 'mixture_pred', 'improvement']
    print(df_final[cols_to_show].head(10))
else:
    print("No synergistic smells found.")
