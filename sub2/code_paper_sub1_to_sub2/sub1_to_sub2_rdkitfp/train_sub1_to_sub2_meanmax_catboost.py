import pandas as pd
from catboost import CatBoostRegressor
import os
import joblib

os.system("rm -rf model*")
# === Paths ===
TRAIN_PATH = "train_sub2tosub1.csv"
MAP_PATH = "../../../sub1/data/raw/TASK1_Stimulus_definition.csv"
FEATURES_PATHS = {
#    "maccs": "../../../sub1/data/processed/features_maccs.csv",
    "rdkitfp": "../../../sub1/data/processed/features_rdkitfp.csv",
#    "morgan": "../../../sub1/data/processed/features_morgan.csv",
#    "descriptors": "../../../sub1/data/processed/features_descriptors.csv"
}
BASE_MODEL_DIR = "models"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# === Load shared data ===
df_train = pd.read_csv(TRAIN_PATH)
df_map = pd.read_csv(MAP_PATH)
df_merged = df_train.merge(df_map[['stimulus', 'molecule']], on='stimulus', how='left')
label_columns = [col for col in df_train.columns if col != 'stimulus']

# === Train models for each feature set ===
for feature_name, feat_path in FEATURES_PATHS.items():
    print(f"\n🔧 Training models on: {feature_name}")

    # Prepare output model dir
    model_dir = os.path.join(BASE_MODEL_DIR, feature_name)
    os.makedirs(model_dir, exist_ok=True)

    df_feats = pd.read_csv(feat_path)
    df_final = df_merged.merge(df_feats, on='molecule', how='left')
    df_final = df_final.drop(columns=['stimulus', 'molecule'])

    # 添加 num_cids 特征，保持和 predict 阶段一致
    df_final['num_cids'] = 1

    feature_columns = [col for col in df_final.columns if col not in label_columns and col != 'SMILES']
    X = df_final[feature_columns]
    y = df_final[label_columns]

    print(f"📌 Feature matrix shape for {feature_name}: {X.shape}")

    for label in label_columns:
        print(f"Training model for: {label} [{feature_name}]")

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.01,
            depth=6,
            random_seed=42,
            verbose=100  # 输出训练进度
        )

        model.fit(X, y[label])

        model_path = os.path.join(model_dir, f"model_{label}_{feature_name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved model: {model_path}")

