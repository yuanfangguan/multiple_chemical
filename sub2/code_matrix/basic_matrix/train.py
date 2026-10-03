import pandas as pd
import os
import pickle
from catboost import CatBoostRegressor

# === Step 1: Load the data ===

# Load training data (descriptors + intensity/pleasantness)
df = pd.read_csv("train.csv")

# Load stimulus to components mapping
stim2comp = pd.read_csv("../../data/raw/TASK2_Stimulus_definition.csv")
stim2comp['components'] = stim2comp['components'].apply(lambda x: list(map(int, x.split(';'))))
stimulus_to_components = dict(zip(stim2comp['id'], stim2comp['components']))

# === Step 2: Generate binary features ===

# Get all unique components across all stimuli
all_components = sorted({comp for comps in stimulus_to_components.values() for comp in comps})

# Function to convert stimulus to binary component feature vector
def get_binary_features(stim_id):
    comps = stimulus_to_components.get(stim_id, [])
    return [1 if c in comps else 0 for c in all_components]

# Apply the function to build the binary feature matrix
X = df['stimulus'].apply(get_binary_features)
X = pd.DataFrame(X.tolist(), columns=[f'comp_{c}' for c in all_components])

# === Step 3: Train one CatBoost model per descriptor ===

# Get target columns (excluding 'stimulus', 'Intensity', 'Pleasantness')
target_columns = df.columns[3:]

# Directory to store models
os.makedirs("models", exist_ok=True)
models = {}

for target in target_columns:
    y = df[target]
    model = CatBoostRegressor(verbose=0, random_seed=42)
    model.fit(X, y)
    model.save_model(f"models/catboost_{target}.cbm")
    print(f"Trained and saved model for descriptor: {target}")

# === Step 4: Save component list (for use in prediction) ===

with open("models/component_list.pkl", "wb") as f:
    pickle.dump(all_components, f)

print("Saved component list.")

