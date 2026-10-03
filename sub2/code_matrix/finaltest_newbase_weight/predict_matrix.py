import pandas as pd
import pickle
import sys
import os
from catboost import CatBoostRegressor

# === Load data ===
df_test = pd.read_csv("test.csv")
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
comp_def = pd.read_csv("../../data/raw/TASK2_Component_definition.csv")

# === Load metadata ===
with open("models_cid_dilution_solvent/cid_list.pkl", "rb") as f:
    all_cids = pickle.load(f)
with open("models_cid_dilution_solvent/solvent_list.pkl", "rb") as f:
    all_solvents = pickle.load(f)

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
for stim_id, comp_ids in stimulus_to_components.items():
    cid_info = {}
    for comp_id in comp_ids:
        if comp_id in component_to_info:
            cid, dilution, solvent = component_to_info[comp_id]
            if cid not in cid_info:
                cid_info[cid] = {'dilution': dilution, 'solvents': set([solvent])}
            else:
                cid_info[cid]['dilution'] = max(cid_info[cid]['dilution'], dilution)
                cid_info[cid]['solvents'].add(solvent)
    stimulus_to_cidinfo[stim_id] = cid_info

# === Build test features ===
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

X_test = df_test['stimulus'].apply(build_features)

feature_names = []
for cid in all_cids:
    feature_names.append(f'cid_{cid}')
    feature_names.append(f'cid_{cid}_dilution')
    for solvent in all_solvents:
        feature_names.append(f'cid_{cid}_{solvent}')

X_test = pd.DataFrame(X_test.tolist(), columns=feature_names)

# === Predict ===
predictions = pd.DataFrame()
predictions['stimulus'] = df_test['stimulus']

model_dir = "models_cid_dilution_solvent"
for fname in os.listdir(model_dir):
    if fname.endswith(".cbm"):
        target = fname.replace("catboost_", "").replace(".cbm", "")
        model = CatBoostRegressor()
        model.load_model(os.path.join(model_dir, fname))
        predictions[target] = model.predict(X_test)
        print(f"Predicted: {target}")

predictions.to_csv("predictions_matrix.csv", index=False)

