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

# Load component list (used in training)
with open("models/component_list.pkl", "rb") as f:
    all_components = pickle.load(f)

# === Step 2: Build binary features for test stimuli ===

def get_binary_features(stim_id):
    comps = stimulus_to_components.get(stim_id, [])
    return [1 if c in comps else 0 for c in all_components]

X_test = df_test['stimulus'].apply(get_binary_features)
X_test = pd.DataFrame(X_test.tolist(), columns=[f'comp_{c}' for c in all_components])

# === Step 3: Load trained models and predict ===

# List of trained model files
model_files = [f for f in os.listdir("models") if f.endswith(".cbm")]

# Store predictions in a new DataFrame
predictions = pd.DataFrame()
predictions["stimulus"] = df_test["stimulus"]

for model_file in model_files:
    descriptor = model_file.replace("catboost_", "").replace(".cbm", "")
    model = CatBoostRegressor()
    model.load_model(os.path.join("models", model_file))
    
    pred = model.predict(X_test)
    predictions[descriptor] = pred
    print(f"Predicted descriptor: {descriptor}")

# === Step 4: Save results ===

predictions.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")

