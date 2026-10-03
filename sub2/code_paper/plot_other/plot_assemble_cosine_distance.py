import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define directories (each contains evaluation.tsv.* directly) ===
assemble_dirs = {
    "Nonparallel": "../new_nonparrallel_noRATA",
    "BaseCatBoost": "../newbase_catboost_noRATA",
    "BaseSepCatBoost": "../newbase_sep_catboost_noRATA"
}

def load_data(folder, assemble_name):
    dfs = []
    for i in range(5):
        path = os.path.join(folder, f"evaluation.tsv.{i}")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            df["assemble"] = assemble_name
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# === Load all data ===
all_data = []
for assemble_name, folder in assemble_dirs.items():
    df = load_data(folder, assemble_name)
    if df is not None:
        all_data.append(df)

df_all = pd.concat(all_data)
df_all = df_all[~df_all["label"].isin(["ALL", "MEAN"])]

# === Plot ===
sns.set_context("talk", font_scale=1.2)
palette = {
    "Nonparallel": "#8B5CF6", 
    "BaseCatBoost": "#EF4444", 
    "BaseSepCatBoost": "#3B82F6"
}

plt.figure(figsize=(8, max(6, len(df_all["label"].unique()) * 0.25)))  # auto height
ax = sns.boxplot(
    data=df_all,
    y="label",          # smell types vertically
    x="cosine",         # Cosine distance horizontally
    hue="assemble",     # compare assemble methods
    palette=palette,
    linewidth=0.8,
    fliersize=2,
    width=0.6,
    dodge=0.6
)

plt.title("Assemble Method Comparison by Smell Type (Cosine Distance)", fontsize=18)
plt.xlabel("Cosine distance", fontsize=14)
plt.ylabel("Smell type", fontsize=14)

# Move legend outside the box
plt.legend(
    title="Assemble method",
    fontsize=11,
    title_fontsize=12,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend on the right
plt.savefig("cosine_distance_by_smell_assemble_methods.png", dpi=300, bbox_inches='tight')
plt.show()

