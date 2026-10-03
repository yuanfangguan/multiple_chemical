import pandas as pd
import os
os.system("rm -rf models*")
import pickle
from catboost import CatBoostRegressor

# === Load data ===
df = pd.read_csv("train.csv")
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
comp_def = pd.read_csv("../../data/raw/TASK2_Component_definition.csv")

# === Build mappings ===
stim2comp['components'] = stim2comp['components'].apply(lambda x: list(map(int, x.split(';'))))
stimulus_to_components = dict(zip(stim2comp['id'], stim2comp['components']))

component_to_info = {}
for _, row in comp_def.iterrows():
    try:
        cid = int(row['CID'])
        dilution = float(row['dilution'])
        solvent = str(row['solvent']).strip().lower()
        component_to_info[int(row['id'])] = (cid, dilution, solvent)
    except:
        continue

stimulus_to_cidinfo = {}
all_cids = set()
all_solvents = set()

for stim_id, comp_ids in stimulus_to_components.items():
    cid_info = {}
    for comp_id in comp_ids:
        if comp_id in component_to_info:
            cid, dilution, solvent = component_to_info[comp_id]
            all_cids.add(cid)
            all_solvents.add(solvent)
            if cid not in cid_info:
                cid_info[cid] = {'dilution': dilution, 'solvents': set([solvent])}
            else:
                cid_info[cid]['dilution'] = max(cid_info[cid]['dilution'], dilution)
                cid_info[cid]['solvents'].add(solvent)
    stimulus_to_cidinfo[stim_id] = cid_info

all_cids = sorted(all_cids)
all_solvents = sorted(all_solvents)

# === Build features ===
def build_features(stim_id):
    cid_dict = stimulus_to_cidinfo.get(stim_id, {})
    features = []
    for cid in all_cids:
        present = cid in cid_dict
        features.append(1 if present else 0)
        features.append(cid_dict[cid]['dilution'] if present else 0.0)
        for solvent in all_solvents:
            features.append(1 if present and solvent in cid_dict[cid]['solvents'] else 0)
    return features

X = df['stimulus'].apply(build_features)

feature_names = []
for cid in all_cids:
    feature_names.append(f'cid_{cid}')
    feature_names.append(f'cid_{cid}_dilution')
    for solvent in all_solvents:
        feature_names.append(f'cid_{cid}_{solvent}')

X = pd.DataFrame(X.tolist(), columns=feature_names)

# === Train models ===
os.makedirs("models_cid_dilution_solvent", exist_ok=True)
target_columns = df.columns[1:]

for target in target_columns:
    y = df[target]
    model = CatBoostRegressor(verbose=0, random_seed=42)
    model.fit(X, y)
    model.save_model(f"models_cid_dilution_solvent/catboost_{target}.cbm")
    print(f"Trained model for {target}")

# === Save metadata ===
with open("models_cid_dilution_solvent/cid_list.pkl", "wb") as f:
    pickle.dump(all_cids, f)
with open("models_cid_dilution_solvent/solvent_list.pkl", "wb") as f:
    pickle.dump(all_solvents, f)

