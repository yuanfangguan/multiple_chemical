import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define directories ===
assemble_dirs = {
    "Nonparallel": "../new_nonparrallel_noRATA",
    "BaseCatBoost": "../newbase_catboost_noRATA",
    "BaseSepCatBoost": "../newbase_sep_catboost_noRATA"
}

def load_data(folder, assemble_name):
    """Load evaluation files and compute mean cosine per fold."""
    fold_means = []
    for i in range(5):
        path = os.path.join(folder, f"evaluation.tsv.{i}")
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            df = df[~df["label"].isin(["ALL", "MEAN"])]  # exclude summary rows
            mean_cosine = df["cosine"].mean()
            fold_means.append({"assemble": assemble_name, "fold": i, "cosine": mean_cosine})
    return pd.DataFrame(fold_means)

# === Load data from all assemble methods ===
df_all = pd.concat([load_data(folder, name) for name, folder in assemble_dirs.items()], ignore_index=True)

# === Plot ===
sns.set_context("talk", font_scale=1.3)
palette = {
    "Nonparallel": "#8B5CF6",   # purple
    "BaseCatBoost": "#EF4444",  # red
    "BaseSepCatBoost": "#3B82F6" # blue
}

plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df_all,
    x="assemble",
    y="cosine",
    palette=palette,
    width=0.5,
    linewidth=0.8
)

# Add individual fold means as dots
sns.stripplot(
    data=df_all,
    x="assemble",
    y="cosine",
    color="black",
    size=6,
    jitter=True,
    alpha=0.7
)

plt.title("Mean Cosine Distance Across Folds", fontsize=18)
plt.xlabel("Assemble method", fontsize=14)
plt.ylabel("Mean cosine distance", fontsize=14)

# Rotate x-axis labels slightly to prevent overlap
plt.xticks(rotation=20, ha='right')

plt.tight_layout()
plt.savefig("mean_cosine_distance_across_folds.png", dpi=300, bbox_inches="tight")
plt.show()

