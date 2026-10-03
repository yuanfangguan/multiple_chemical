import pandas as pd
import os
import pickle
from catboost import CatBoostRegressor

# === Step 1: Load CSVs ===

df = pd.read_csv("train.csv")
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
comp2cid = pd.read_csv("../../data/raw/TASK2_Component_definition.csv")

# === Step 2: Build mappings ===

# stimulus -> list of component IDs
stim2comp['components'] = stim2comp['components'].apply(lambda x: list(map(int, x.split(';'))))
stimulus_to_components = dict(zip(stim2comp['id'], stim2comp['components']))

# component ID -> CID (use first match if duplicated)
component_to_cid = dict(zip(comp2cid['id'], comp2cid['CID']))

# === Step 3: Map each stimulus to set of CIDs ===

stimulus_to_cids = {}
for stim_id, comps in stimulus_to_components.items():
    cids = set(component_to_cid[c] for c in comps if c in component_to_cid)
    stimulus_to_cids[stim_id] = list(cids)

# Get all unique CIDs
all_cids = sorted({cid for cid_list in stimulus_to_cids.values() for cid in cid_list})

# === Step 4: Create binary CID features ===

def get_cid_features(stim_id):
    cids = stimulus_to_cids.get(stim_id, [])
    return [1 if cid in cids else 0 for cid in all_cids]

X = df['stimulus'].apply(get_cid_features)
X = pd.DataFrame(X.tolist(), columns=[f'cid_{cid}' for cid in all_cids])

# === Step 5: Train one CatBoost model per descriptor ===

target_columns = df.columns[3:]  # skip 'stimulus', 'Intensity', 'Pleasantness'

os.makedirs("models_cid", exist_ok=True)

for target in target_columns:
    y = df[target]
    model = CatBoostRegressor(verbose=0, random_seed=42)
    model.fit(X, y)
    model.save_model(f"models_cid/catboost_cid_{target}.cbm")
    print(f"Trained and saved CID-based model for descriptor: {target}")

# === Step 6: Save CID list for prediction ===

with open("models_cid/cid_list.pkl", "wb") as f:
    pickle.dump(all_cids, f)

print("Saved all_cids to models_cid/cid_list.pkl")

