import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define directories ===
cat_dirs = {
    "Descriptors": "../new_descriptors",
    "MACCS": "../new_maccs",
    "Mordred": "../new_mordred",
    "Morgan": "../new_morgan",
    "RDKitFP": "../new_rdkitfp"
}

xgb_dirs = {
    "Descriptors": "../../code_paper_xgboost/new_descriptors",
    "MACCS": "../../code_paper_xgboost/new_maccs",
    "Mordred": "../../code_paper_xgboost/new_mordred",
    "Morgan": "../../code_paper_xgboost/new_morgan",
    "RDKitFP": "../../code_paper_xgboost/new_rdkitfp"
}

lgb_dirs = {
    "Descriptors": "../../code_paper_lightgbm/new_descriptors",
    "MACCS": "../../code_paper_lightgbm/new_maccs",
    "Mordred": "../../code_paper_lightgbm/new_mordred",
    "Morgan": "../../code_paper_lightgbm/new_morgan",
    "RDKitFP": "../../code_paper_lightgbm/new_rdkitfp"
}

def load_data(folder, learner_name):
    dfs = []
    for i in range(5):
        path = os.path.join(folder, f"evaluation.tsv.{i}")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            df["learner"] = learner_name
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# === Load all data ===
all_data = []
for name in cat_dirs:
    df_cat = load_data(cat_dirs[name], "CatBoost")
    df_xgb = load_data(xgb_dirs[name], "XGBoost")
    df_lgb = load_data(lgb_dirs[name], "LightGBM")

    for df in [df_cat, df_xgb, df_lgb]:
        if df is not None:
            df["feature"] = name
            all_data.append(df)

df_all = pd.concat(all_data)
df_all = df_all[~df_all["label"].isin(["ALL", "MEAN"])]

# === Plot settings ===
sns.set_context("talk", font_scale=1.1)
palette = {"CatBoost": "#8B5CF6", "XGBoost": "#EF4444", "LightGBM": "#3B82F6"}

# === Generate one vertical plot per feature ===
for feature_name in cat_dirs.keys():
    df_feature = df_all[df_all["feature"] == feature_name]

    plt.figure(figsize=(7, max(6, len(df_feature["label"].unique()) * 0.25)))  # auto height
    sns.boxplot(
        data=df_feature,
        y="label",             # smell types vertically
        x="cosine",            # cosine distance horizontally
        hue="learner",         # 3 base learners
        palette=palette,
        linewidth=0.8,
        fliersize=2,
        width=0.6,
        dodge=0.6
    )

    plt.title(f"Cosine Distance by Smell Type ({feature_name})", fontsize=18)
    plt.xlabel("Cosine distance", fontsize=14)
    plt.ylabel("Smell type", fontsize=14)
    plt.legend(title="Base learner", fontsize=11, title_fontsize=12, loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(f"cosine_distance_by_smell_{feature_name}_learners.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✅ All 5 vertical cosine plots saved (one per feature group, comparing learners).")

