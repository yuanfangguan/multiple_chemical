import pandas as pd
import pickle
import os
from catboost import CatBoostRegressor

# === Step 1: Load test data and mappings ===

# Load test data
df_test = pd.read_csv("test.csv")

# Load stimulus-to-component mapping
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
stim2comp['components'] = stim2comp['components'].apply(lambda x: list(map(int, x.split(';'))))
stimulus_to_components = dict(zip(stim2comp['id'], stim2comp['components']))

# Load component-to-CID mapping
comp2cid = pd.read_csv("../../data/raw/TASK2_Component_definition.csv")
component_to_cid = dict(zip(comp2cid['id'], comp2cid['CID']))

# Load list of all CIDs used during training
with open("models_cid/cid_list.pkl", "rb") as f:
    all_cids = pickle.load(f)

# === Step 2: Build binary CID features for test set ===

def get_cid_features(stim_id):
    comps = stimulus_to_components.get(stim_id, [])
    cids = set(component_to_cid[c] for c in comps if c in component_to_cid)
    return [1 if cid in cids else 0 for cid in all_cids]

X_test = df_test['stimulus'].apply(get_cid_features)
X_test = pd.DataFrame(X_test.tolist(), columns=[f'cid_{cid}' for cid in all_cids])

# === Step 3: Load trained models and predict each descriptor ===

model_dir = "models_cid"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".cbm")]

predictions = pd.DataFrame()
predictions["stimulus"] = df_test["stimulus"]

for model_file in model_files:
    descriptor = model_file.replace("catboost_cid_", "").replace(".cbm", "")
    model = CatBoostRegressor()
    model.load_model(os.path.join(model_dir, model_file))

    pred = model.predict(X_test)
    predictions[descriptor] = pred
    print(f"Predicted: {descriptor}")

# === Step 4: Save prediction results ===

predictions.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions_cid.csv")

