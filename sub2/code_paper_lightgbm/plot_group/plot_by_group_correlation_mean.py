import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set global font size for all plot elements
plt.rcParams.update({'font.size': 20})

# === Define model directories (Updated to your 6 categories) ===
model_dirs = {
    "Mean": "../new_nonparrallel_noRATA_eva_by_number_meanonly/",
    "Max": "../new_nonparrallel_noRATA_eva_by_number_maxonly/",
    "Mean+Max": "../new_nonparrallel_noRATA_eva_by_number/",
    "Softmax": "../../code_paper_A_lightgbm/new_nonparrallel_noRATA_eva_by_number/",
    "Gated": "../../code_paper_B_lightgbm/new_nonparrallel_noRATA_eva_by_num/",
    "Single": "../newbase_sep_catboost_noRATA_eva_by_number/"
}

def load_eval_data(base_dir, model_label):
    """Traverse folds and evaluation files, compute mean Pearson correlation."""
    results = []
    # Search for directories like evaluation_groups_0, evaluation_groups_1, etc.
    fold_dirs = sorted(glob.glob(os.path.join(base_dir, "evaluation_groups_*")))
    
    for fold_dir in fold_dirs:
        fold_num = os.path.basename(fold_dir).split("_")[-1]
        # Search for evaluation_numcomp_1.tsv, etc.
        for file in glob.glob(os.path.join(fold_dir, "evaluation_*.tsv")):
            try:
                df = pd.read_csv(file, sep="\t")
                if "pearson" not in df.columns:
                    continue
                
                # We want the MEAN across all odors for this specific group (num chemicals)
                # If your file has an 'ALL' or 'MEAN' row, use it; otherwise, average the column
                if "label" in df.columns and (df["label"] == "MEAN").any():
                    mean_pearson = df.loc[df["label"] == "MEAN", "pearson"].values[0]
                else:
                    mean_pearson = df["pearson"].dropna().mean()

                group_name = os.path.splitext(os.path.basename(file))[0].replace("evaluation_", "")
                
                results.append({
                    "model": model_label,
                    "fold": fold_num,
                    "group": group_name,
                    "pearson": mean_pearson
                })
            except Exception as e:
                print(f"Skipping {file}: {e}")
    return pd.DataFrame(results)

# === Load and Process Data ===
all_data = []
for label, folder in model_dirs.items():
    if os.path.exists(folder):
        df = load_eval_data(folder, label)
        if not df.empty:
            all_data.append(df)
    else:
        print(f"⚠️ Directory not found: {folder}")

if not all_data:
    print("No data loaded. Please check your directory paths.")
    exit()

df_all = pd.concat(all_data, ignore_index=True)


# === Clean and Sort Groups (1, 2, ..., all) ===
def sort_key(x):
    if x == "all": return 999
    try:
        return int(x.replace("numcomp_", ""))
    except:
        return 999

df_all["group"] = df_all["group"].apply(lambda x: x.replace("numcomp_", "") if "numcomp_" in x else x)
df_all = df_all[df_all["group"] != "all"]
unique_groups = sorted(df_all["group"].unique(), key=sort_key)

# === Plotting ===
plt.figure(figsize=(12, 10))

# Use a palette that distinguishes the 6 methods well
palette = "Set2" 

ax = sns.boxplot(
    data=df_all,
    x="pearson",
    y="group",
    hue="model",
    order=unique_groups,
    palette=palette,
    linewidth=1.2,
    width=0.8,
    dodge=True,
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"}
)

# Text and Labels
plt.xlabel("Mean Pearson correlation (r)", fontsize=20)
plt.ylabel("Number of chemicals per stimulus", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Adjust legend to accommodate 6 items
plt.legend(
    title="Pooling Strategy (with LightGBM)",
    fontsize=20,
    title_fontsize=20,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True
)

plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

# Save the figure
plt.savefig("pearson_by_chemical_number_6groups.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Saved: pearson_by_chemical_number_6groups.png")
