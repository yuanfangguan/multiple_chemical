import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define strategy directories ===
method_dirs = {
    "Mean": "../code_paper_lightgbm/new_nonparrallel_noRATA_meanonly/",
    "Max": "../code_paper_lightgbm/new_nonparrallel_noRATA_maxonly/",
    "Mean+Max": "../code_paper_lightgbm/new_nonparrallel_noRATA/",
    "Softmax": "../code_paper_A_lightgbm/new_nonparrallel_noRATA/",
    "Gated": "../code_paper_B_lightgbm/new_nonparrallel_noRATA/",
    "Single": "../code_paper_lightgbm/newbase_sep_catboost_noRATA/"
}

records = []

# === Read all folds and keep all percepts (exclude ALL/MEAN for now) ===
for method, path in method_dirs.items():
    for i in range(5):
        eval_file = os.path.join(path, f"evaluation.tsv.{i}")
        if not os.path.exists(eval_file):
            print(f"⚠️ Missing: {eval_file}")
            continue
        
        try:
            df = pd.read_csv(eval_file, sep="\t")
            # Filter out aggregate rows
            df = df[~df["label"].isin(["ALL", "MEAN"])].copy()
            df["fold"] = i
            df["method"] = method
            records.append(df)
        except Exception as e:
            print(f"❌ Error reading {eval_file}: {e}")

if not records:
    print("No data found! Please check your file paths.")
    exit()

df_all = pd.concat(records, ignore_index=True)

# === Compute average performance per percept per method ===
df_avg = (
    df_all.groupby(["label", "method"])
    [["pearson", "cosine"]]
    .mean()
    .reset_index()
)

# === Determine sorting order based on the 'Mean+Max' strategy ===
# FIX: The string here must match the key in method_dirs exactly
baseline_name = "Mean+Max" 
baseline_means = (
    df_avg[df_avg["method"] == baseline_name]
    .sort_values("pearson", ascending=False)
)

odor_order = baseline_means["label"].tolist()

# Defensive check: If "Mean+Max" wasn't found, fallback to alphabetical order
if not odor_order:
    print(f"⚠️ Warning: Strategy '{baseline_name}' not found in data. Using default order.")
    odor_order = sorted(df_all["label"].unique())

# === Plot: Pearson correlation per percept ===
plt.figure(figsize=(10, 22)) # Slightly wider for better readability
sns.boxplot(
    data=df_all,
    y="label",
    x="pearson",
    hue="method",
    order=odor_order,
    palette="Set2",
    orient="h",
    fliersize=2  # Smaller outlier dots
)
plt.title("Predictive Performance per Odor Descriptor (Pearson r) with LightGBM", fontsize=14)
plt.xlabel("Pearson Correlation")
plt.ylabel("Odor Descriptor")
plt.grid(axis='x', linestyle='--', alpha=0.6) # Add grid for easier reading
plt.legend(title="Pooling Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("per_percept_pearson_vertical.png", dpi=300)
plt.close()

# === Plot: Cosine distance per percept ===
plt.figure(figsize=(10, 22))
sns.boxplot(
    data=df_all,
    y="label",
    x="cosine",
    hue="method",
    order=odor_order,
    palette="Set2",
    orient="h",
    fliersize=2
)
plt.title("Predictive Performance per Odor Descriptor (Cosine Distance)", fontsize=14)
plt.xlabel("Cosine Distance (Lower = Better)")
plt.ylabel("Odor Descriptor")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.legend(title="Pooling Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("per_percept_cosine_vertical.png", dpi=300)
plt.close()

print(f"✅ Successfully processed {len(df_all)} records.")
print("✅ Saved: per_percept_pearson_vertical.png and per_percept_cosine_vertical.png")
