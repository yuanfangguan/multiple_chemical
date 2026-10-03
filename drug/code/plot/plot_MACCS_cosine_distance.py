import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Same color map as correlation plot
MODEL_COLORS = {
    "Two-drug": "#66c2a5",
    "Single molecule": "#fc8d62",
    "Mean": "#8da0cb",
    "Max": "#e78ac3",
    "Max+Mean": "#a6d854"
}

base_dir = "../"
models = {
    "Two-drug": "study_twodrug_MACCS_celllineid",
    "Single molecule": "study_singledrug_MACCS_celllineid",
    "Mean": "study_mean_MACCS_celllineid",
    "Max": "study_max_MACCS_celllineid",
    "Max+Mean": "study_maxandmean_MACCS_celllineid",
}

all_results = []

print("🔍 Searching for result files...")
for model_name, folder in models.items():
    for fold in range(5):
        result_path = os.path.join(base_dir, folder, f"results_{fold}", "evaluation_summary_by_study.csv")
        result_path_1 = os.path.join(base_dir, folder, f"results_{fold}", "evaluation_pairlevel_summary.csv")
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            df["model"] = model_name
            df["fold"] = fold
            all_results.append(df)
        elif os.path.exists(result_path_1):
            df = pd.read_csv(result_path_1)
            df["model"] = model_name
            df["fold"] = fold
            all_results.append(df)

if not all_results:
    raise RuntimeError("❌ No evaluation_summary_by_study.csv files found!")

df_all = pd.concat(all_results, ignore_index=True)
df_plot = df_all[["target", "model", "fold", "mean_test_cos_by_study"]].copy()
df_plot = df_plot.rename(columns={"mean_test_cos_by_study": "cosine_distance"})

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df_plot,
    x="target",
    y="cosine_distance",
    hue="model",
    palette=MODEL_COLORS,
    linewidth=1.3,
    fliersize=3
)

plt.ylabel("Cosine Distance Across Studies for MACCS (Lower = Better)", fontsize=13)
plt.xlabel("")
plt.xticks(rotation=30, ha="right", fontsize=12)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=11, title_fontsize=12)
plt.tight_layout()

os.makedirs("results_MACCS", exist_ok=True)
fig_path = "results_MACCS/cosine_comparison_boxplot.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"✅ Saved cosine figure to {fig_path}")

