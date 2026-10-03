import pandas as pd
import pickle
import os
from catboost import CatBoostRegressor

# === Load data and mappings ===

df_test = pd.read_csv("test.csv")
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
comp_def = pd.read_csv("../../data/raw/TASK2_Component_definition.csv")

with open("models_cid_dilution/cid_list.pkl", "rb") as f:
    all_cids = pickle.load(f)

# Build mappings
stim2comp['components'] = stim2comp['components'].apply(lambda x: list(map(int, x.split(';'))))
stimulus_to_components = dict(zip(stim2comp['id'], stim2comp['components']))

component_to_cid_dilution = {}
for _, row in comp_def.iterrows():
    try:
        cid = int(row['CID'])
        dilution = float(row['dilution'])
        component_to_cid_dilution[int(row['id'])] = (cid, dilution)
    except:
        continue

stimulus_to_cidinfo = {}
for stim_id, comp_ids in stimulus_to_components.items():
    cid_dilution = {}
    for comp_id in comp_ids:
        if comp_id in component_to_cid_dilution:
            cid, dilution = component_to_cid_dilution[comp_id]
            if cid in cid_dilution:
                cid_dilution[cid] = max(cid_dilution[cid], dilution)
            else:
                cid_dilution[cid] = dilution
    stimulus_to_cidinfo[stim_id] = cid_dilution

# Build features
def build_features(stim_id):
    cid_dilution = stimulus_to_cidinfo.get(stim_id, {})
    features = []
    for cid in all_cids:
        features.append(1 if cid in cid_dilution else 0)
        features.append(cid_dilution.get(cid, 0.0))
    return features

X_test = df_test['stimulus'].apply(build_features)

feature_names = []
for cid in all_cids:
    feature_names.append(f'cid_{cid}')
    feature_names.append(f'cid_{cid}_dilution')

X_test = pd.DataFrame(X_test.tolist(), columns=feature_names)

# === Predict ===

predictions = pd.DataFrame()
predictions['stimulus'] = df_test['stimulus']

for fname in os.listdir("models_cid_dilution"):
    if fname.endswith(".cbm"):
        target = fname.replace("catboost_", "").replace(".cbm", "")
        model = CatBoostRegressor()
        model.load_model(os.path.join("models_cid_dilution", fname))
        pred = model.predict(X_test)
        predictions[target] = pred
        print(f"Predicted {target}")

predictions.to_csv("predictions.csv", index=False)
print("Saved to predictions_cid_dilution.csv")

