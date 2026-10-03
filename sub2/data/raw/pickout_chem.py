import pandas as pd
import sys

# === Paths ===
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_STRUCTURE_PATH = "CID.csv"

def get_mixture_structures(target_stimulus_id):
    # 1. Load the data with correct headers
    df_stim = pd.read_csv(STIMULUS_MAP_PATH)
    df_comp = pd.read_csv(COMPONENT_MAP_PATH)
    df_cid = pd.read_csv(CID_STRUCTURE_PATH)

    # 2. Create Mappings
    # Map 'id' -> 'CID' from Component definition
    comp_to_cid = dict(zip(df_comp['id'], df_comp['CID']))
    
    # Map 'molecule' -> 'SMILES' from CID.csv
    # We use 'molecule' because that is the header in your CID.csv
    cid_to_smiles = dict(zip(df_cid['molecule'], df_cid['SMILES']))

    # 3. Find the components for the specific mixture
    # We cast to string to handle IDs like 'AK432' vs '166'
    target_stimulus_id = str(target_stimulus_id)
    stim_row = df_stim[df_stim['id'].astype(str) == target_stimulus_id]
    
    if stim_row.empty:
        return f"Stimulus {target_stimulus_id} not found in {STIMULUS_MAP_PATH}"

    # Get the components string and split by semicolon
    components_str = str(stim_row.iloc[0]['components'])
    component_ids = [int(c.strip()) for c in components_str.split(';') if c.strip().replace('-','').isdigit()]

    # 4. Pull the structures
    results = []
    for comp_id in component_ids:
        cid = comp_to_cid.get(comp_id)
        
        smiles = "Structure not found"
        if cid is not None:
            # Handle the SMILES lookup (SMILES might be NaN for solvents)
            raw_smiles = cid_to_smiles.get(cid)
            if pd.isna(raw_smiles) or raw_smiles == "":
                smiles = "No SMILES (likely solvent/oil)"
            else:
                smiles = raw_smiles
        
        results.append({
            'component_id': comp_id,
            'CID': cid,
            'SMILES': smiles
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pickout_chem.py AK432")
    else:
        target = sys.argv[1]
        df_results = get_mixture_structures(target)
        
        if isinstance(df_results, str):
            print(df_results)
        else:
            print(f"\nChemical Structures for Stimulus: {target}")
            print("=" * 60)
            print(df_results.to_string(index=False))
