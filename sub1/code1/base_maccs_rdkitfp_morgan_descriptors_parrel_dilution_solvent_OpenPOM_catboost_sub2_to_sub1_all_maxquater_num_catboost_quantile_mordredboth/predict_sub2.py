import pandas as pd
import os
import joblib
import numpy as np

# === Paths ===
TEST_PATH = "test.csv"
MAP_PATH = "../../data/raw/TASK1_Stimulus_definition.csv"
FEATURES_DIR = "../../data/processed"
FEATURE_FILES = [
    "features_maccs.csv",
    "features_morgan.csv",
    "features_rdkitfp.csv",
    "features_descriptors.csv",
     "../../../sub1/data/raw/Mordred_Descriptors.csv"
]
MODEL_DIR = "models"
OUTPUT_CSV = "predictions_sub2.csv"

AGG_METHODS = ["max", "q75"]  # Consistent with training

# === Load Base Data ===
df_test = pd.read_csv(TEST_PATH)
df_map = pd.read_csv(MAP_PATH)

stimulus_to_molecule = dict(zip(df_map['stimulus'], df_map['molecule']))
stimulus_ids = df_test['stimulus']
label_columns = [col for col in df_test.columns if col != 'stimulus']

# === Store all predictions in: {label: list of arrays from different features and aggregation methods} ===
predictions_per_label = {label: [] for label in label_columns}

for feature_file in FEATURE_FILES:
    print(f"🔄 Processing feature file: {feature_file}")
    full_path = os.path.join(FEATURES_DIR, feature_file)

    if "Mordred_Descriptors.csv" in feature_file:
        df_feats = pd.read_csv(full_path, encoding='ISO-8859-1')
    else:
        df_feats = pd.read_csv(full_path)


    if 'SMILES' in df_feats.columns:
        df_feats = df_feats.drop(columns=['SMILES'])

    df_feats = df_feats.set_index('molecule')

    feature_matrix = []
    for stim in stimulus_ids:
        mol = stimulus_to_molecule.get(stim)
        if mol in df_feats.index:
            feat_vec = df_feats.loc[mol].values
            num_cids = 1
            combined_feat = list(feat_vec) + [num_cids]
            feature_matrix.append(combined_feat)
        else:
            feature_matrix.append([0] * (df_feats.shape[1] + 1))

    X_test = pd.DataFrame(feature_matrix, columns=df_feats.columns.tolist() + ['num_cids'])

    # === Predict for each aggregation method ===
    for agg_name in AGG_METHODS:
        for label in label_columns:
            model_filename = f"model_{label}_{feature_file.replace('.csv', '')}_{agg_name}.pkl"
            model_path = os.path.join(MODEL_DIR, model_filename)

            if os.path.exists(model_path):
                model = joblib.load(model_path)
                preds = model.predict(X_test)
                predictions_per_label[label].append(preds)
            else:
                print(f"⚠️ Model not found: {model_filename}")

# === Assemble predictions per label ===
final_predictions = pd.DataFrame({'stimulus': stimulus_ids})

for label in label_columns:
    if predictions_per_label[label]:
        all_preds = np.vstack(predictions_per_label[label])  # shape: (num_models, num_samples)
        assembled_preds = np.mean(all_preds, axis=0)  # Mean across models
        final_predictions[label] = assembled_preds
    else:
        print(f"⚠️ No predictions collected for label: {label}")

# === Save final assembled predictions ===
final_predictions.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Assembled predictions saved to {OUTPUT_CSV}")

