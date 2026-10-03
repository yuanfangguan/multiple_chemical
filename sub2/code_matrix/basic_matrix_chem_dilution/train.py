import pandas as pd
import os
import pickle
from catboost import CatBoostRegressor

# === Step 1: Load data ===
df = pd.read_csv("train.csv")
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
comp_def = pd.read_csv("../../data/raw/TASK2_Component_definition.csv")

# === Step 2: Build mappings ===

# Stimulus → component list
stim2comp['components'] = stim2comp['components'].apply(lambda x: list(map(int, x.split(';'))))
stimulus_to_components = dict(zip(stim2comp['id'], stim2comp['components']))

# Component ID → (CID, dilution)
component_to_cid_dilution = {}
for _, row in comp_def.iterrows():
    try:
        cid = int(row['CID'])
        dilution = float(row['dilution'])
        component_to_cid_dilution[int(row['id'])] = (cid, dilution)
    except:
        continue  # skip malformed rows

# Stimulus → CID → max dilution
stimulus_to_cidinfo = {}
for stim_id, comp_ids in stimulus_to_components.items():
    cid_dilution = {}
    for cid_ in comp_ids:
        if cid_ in component_to_cid_dilution:
            cid, dilution = component_to_cid_dilution[cid_]
            if cid in cid_dilution:
                cid_dilution[cid] = max(cid_dilution[cid], dilution)
            else:
                cid_dilution[cid] = dilution
    stimulus_to_cidinfo[stim_id] = cid_dilution

# All unique CIDs
all_cids = sorted({cid for cidinfo in stimulus_to_cidinfo.values() for cid in cidinfo})

# === Step 3: Build features ===

def build_features(stim_id):
    cid_dilution = stimulus_to_cidinfo.get(stim_id, {})
    features = []
    for cid in all_cids:
        features.append(1 if cid in cid_dilution else 0)  # binary feature
        features.append(cid_dilution.get(cid, 0.0))        # dilution feature
    return features

X = df['stimulus'].apply(build_features)

# Create feature names
feature_names = []
for cid in all_cids:
    feature_names.append(f'cid_{cid}')
    feature_names.append(f'cid_{cid}_dilution')

X = pd.DataFrame(X.tolist(), columns=feature_names)

# === Step 4: Train models ===

target_columns = df.columns[3:]  # exclude stimulus, intensity, pleasantness
os.makedirs("models_cid_dilution", exist_ok=True)

for target in target_columns:
    y = df[target]
    model = CatBoostRegressor(verbose=0, random_seed=42)
    model.fit(X, y)
    model.save_model(f"models_cid_dilution/catboost_{target}.cbm")
    print(f"Trained model for {target}")

# === Step 5: Save CID list ===

with open("models_cid_dilution/cid_list.pkl", "wb") as f:
    pickle.dump(all_cids, f)

