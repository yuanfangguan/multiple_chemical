import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# CONFIG
# ---------------------------------------
feature_dirs = [
    "../code_maccs",
    "../code_morgan",
    "../code_rdkidfp",
    "../code_descriptors"
]

output_dir = "./metric_plots"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# LOAD ALL RESULTS (ALL METRICS)
# ---------------------------------------
records = []

for fdir in feature_dirs:
    feature_name = os.path.basename(fdir).replace("code_", "")

    for fname in os.listdir(fdir):
        if fname.startswith("evaluation_summary.csv"):
            df = pd.read_csv(os.path.join(fdir, fname))

            # df has multiple rows, each with Metric + Value
            for _, row in df.iterrows():
                records.append({
                    "Feature Set": feature_name,
                    "Metric": row["Metric"],
                    "Value": row["Value"]
                })

results = pd.DataFrame(records)
print("Parsed results:")
print(results.head())

# Get all metrics present
metrics = results["Metric"].unique()

# ---------------------------------------
# PLOT + SAVE
# ---------------------------------------
sns.set(style="whitegrid", font_scale=1.2)

for metric in metrics:
    plt.figure(figsize=(8, 5))
    df_plot = results[results["Metric"] == metric]

    sns.boxplot(data=df_plot, x="Feature Set", y="Value")
    sns.stripplot(data=df_plot, x="Feature Set", y="Value",
                  color='black', size=4, jitter=True)

    plt.title(f"{metric} Comparison Across Feature Sets")
    plt.xlabel("Feature Set")
    plt.ylabel(metric)
    plt.tight_layout()

    # save file with safe filename
    clean_name = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    save_path = os.path.join(output_dir, f"{clean_name}.png")

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

