import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# === Define model directories and labels ===
model_dirs = {
    "Non-parallel assemble": "../new_nonparrallel_noRATA_eva_by_number/",
    "Non-parallel assemble (Max)": "../new_nonparrallel_noRATA_eva_by_number_maxonly/",
    "Non-parallel assemble (Avg)": "../new_nonparrallel_noRATA_eva_by_number_meanonly/",
    "Single chemical model": "../newbase_sep_catboost_noRATA_eva_by_number/"
}

def load_eval_data(base_dir, model_label):
    """Traverse folds and evaluation files, compute per-smell mean Pearson ignoring NaN."""
    results = []
    for fold_dir in sorted(glob.glob(os.path.join(base_dir, "evaluation_groups_*"))):
        fold_num = os.path.basename(fold_dir).split("_")[-1]
        for file in glob.glob(os.path.join(fold_dir, "evaluation_*.tsv")):
            try:
                df = pd.read_csv(file, sep="\t")
                if "pearson" not in df.columns or "label" not in df.columns:
                    continue
                df = df[~df["label"].isin(["ALL", "MEAN"])]  # drop summaries
                df = df.dropna(subset=["pearson"])
                df["model"] = model_label
                df["fold"] = fold_num
                df["group"] = os.path.splitext(os.path.basename(file))[0].replace("evaluation_", "")
                results.append(df[["label", "pearson", "model", "fold", "group"]])
            except Exception as e:
                print(f"Skipping {file}: {e}")
                continue
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

# === Load data for all models ===
all_data = []
for label, folder in model_dirs.items():
    df = load_eval_data(folder, label)
    if not df.empty:
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# === Plot ===
sns.set_context("talk", font_scale=1.2)
palette = {
    "Non-parallel assemble": "#8B5CF6",
    "Non-parallel assemble (Max)": "#A855F7",
    "Non-parallel assemble (Avg)": "#C084FC",
    "Single chemical model": "#3B82F6"
}

plt.figure(figsize=(10, max(6, len(df_all["label"].unique()) * 0.25)))
sns.boxplot(
    data=df_all,
    x="pearson",
    y="label",
    hue="model",
    palette=palette,
    linewidth=0.8,
    width=0.6,
    dodge=0.6
)

plt.title("Mean Pearson Correlation per Smell Type", fontsize=18)
plt.xlabel("Pearson correlation", fontsize=14)
plt.ylabel("Smell type", fontsize=14)

# Move legend outside the figure
plt.legend(
    title="Model",
    fontsize=11,
    title_fontsize=12,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("pearson_correlation_per_smelltype.png", dpi=300, bbox_inches="tight")
plt.show()

