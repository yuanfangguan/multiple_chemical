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
    """Traverse folds and evaluation files, compute mean cosine distance ignoring NaN."""
    results = []
    for fold_dir in sorted(glob.glob(os.path.join(base_dir, "evaluation_groups_*"))):
        fold_num = os.path.basename(fold_dir).split("_")[-1]
        for file in glob.glob(os.path.join(fold_dir, "evaluation_*.tsv")):
            try:
                df = pd.read_csv(file, sep="\t")
                if "cosine" not in df.columns:
                    continue
                mean_cosine = df["cosine"].dropna().mean()
                group_name = os.path.splitext(os.path.basename(file))[0].replace("evaluation_", "")
                results.append({
                    "model": model_label,
                    "fold": fold_num,
                    "group": group_name,
                    "cosine": mean_cosine
                })
            except Exception as e:
                print(f"Skipping {file}: {e}")
                continue
    return pd.DataFrame(results)

# === Load all model data ===
all_data = []
for label, folder in model_dirs.items():
    df = load_eval_data(folder, label)
    if not df.empty:
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# === Clean and order groups ===
def sort_key(x):
    if x == "all":
        return 999
    try:
        return int(x.replace("numcomp_", ""))
    except:
        return 999

df_all["group"] = df_all["group"].apply(lambda x: x.replace("numcomp_", "") if "numcomp_" in x else x)
df_all = df_all.sort_values(by="group", key=lambda x: x.map(sort_key))

# === Plot ===
sns.set_context("talk", font_scale=1.2)
palette = {
    "Non-parallel assemble": "#8B5CF6",
    "Non-parallel assemble (Max)": "#A855F7",
    "Non-parallel assemble (Avg)": "#C084FC",
    "Single chemical model": "#3B82F6"
}

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_all,
    x="cosine",
    y="group",
    hue="model",
    palette=palette,
    linewidth=0.8,
    width=0.6,
    dodge=0.6
)

plt.title("Mean Cosine Distance by Number of Chemicals per Stimulus", fontsize=18)
plt.xlabel("Mean cosine distance", fontsize=14)
plt.ylabel("Number of chemicals per stimulus", fontsize=14)

# Legend outside to avoid overlap
plt.legend(
    title="Model",
    fontsize=11,
    title_fontsize=12,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("mean_cosine_by_numchemicals_horizontal.png", dpi=300, bbox_inches="tight")
plt.show()

