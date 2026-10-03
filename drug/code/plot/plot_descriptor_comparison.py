import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Unified color map (same for both plots)
MODEL_COLORS = {
    "Two-drug": "#66c2a5",
    "Single molecule": "#fc8d62",
    "Mean": "#8da0cb",
    "Max": "#e78ac3",
    "Max+Mean": "#a6d854"
}

# ========================
# Paths
# ========================
base_dir = "../"
models = {
    "Two-drug": "study_twodrug_descriptor_celllineid",
    "Single molecule": "study_singledrug_descriptor_celllineid",
    "Mean": "study_mean_descriptor_celllineid",
    "Max": "study_max_descriptor_celllineid",
    "Max+Mean": "study_maxandmean_descriptor_celllineid",
}

# ========================
# Collect results
# ========================
all_results = []

print("🔍 Searching for result files...")
for model_name, folder in models.items():
    for fold in range(5):
        result_path = os.path.join(base_dir, folder, f"results_{fold}", "evaluation_summary_by_study.csv")
        result_path_1 = os.path.join(base_dir, folder, f"results_{fold}", "evaluation_pairlevel_summary.csv")
        if os.path.exists(result_path):
            print(f"✅ Found: {result_path}")
            df = pd.read_csv(result_path)
            df["model"] = model_name
            df["fold"] = fold
            all_results.append(df)
        elif os.path.exists(result_path_1):
            print(f"✅ Found: {result_path}")
            df = pd.read_csv(result_path_1)
            df["model"] = model_name
            df["fold"] = fold
            all_results.append(df)
        else:
            print(f"⚠️ Missing: {result_path}")

if not all_results:
    raise RuntimeError("❌ No evaluation_summary_by_study.csv files found!")

df_all = pd.concat(all_results, ignore_index=True)

# ========================
# Prepare data for plotting
# ========================
df_plot = df_all[["target", "model", "fold", "mean_test_r_by_study"]].copy()
df_plot = df_plot.rename(columns={"mean_test_r_by_study": "pearson_r"})

# ========================
# Plot
# ========================
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df_plot,
    x="target",
    y="pearson_r",
    hue="model",
    palette=MODEL_COLORS,
    linewidth=1.3,
    fliersize=3
)

plt.ylabel("Pearson Correlation Across Studies for descriptor", fontsize=13)
plt.xlabel("")
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11, title_fontsize=12)
plt.tight_layout()

os.makedirs("results_descriptor", exist_ok=True)
fig_path = "results_descriptor/pearson_comparison_boxplot.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"✅ Saved correlation figure to {fig_path}")

