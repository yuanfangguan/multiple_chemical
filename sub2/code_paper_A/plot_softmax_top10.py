import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global font size for all plot elements
plt.rcParams.update({'font.size': 14})

# === Configuration ===
method_dirs = {
    "Mean": "../code_paper/new_nonparrallel_noRATA_meanonly/",
    "Max": "../code_paper/new_nonparrallel_noRATA_maxonly/",
    "Mean+Max": "../code_paper/new_nonparrallel_noRATA/",
    "Softmax": "../code_paper_A/new_nonparrallel_noRATA/",
    "Gated": "../code_paper_B/new_nonparrallel_noRATA/",
    "Single": "../code_paper/newbase_sep_catboost_noRATA/"
}

records = []

# === Data Loading ===
for method, path in method_dirs.items():
    for i in range(5):
        eval_file = os.path.join(path, f"evaluation.tsv.{i}")
        if os.path.exists(eval_file):
            try:
                df = pd.read_csv(eval_file, sep="\t")
                df = df[~df["label"].isin(["ALL", "MEAN"])].copy()
                df["method"] = method
                records.append(df)
            except Exception as e:
                print(f"Error reading {eval_file}: {e}")

if not records:
    print("No data found. Check your file paths.")
    exit()

df_all = pd.concat(records, ignore_index=True)

# === Analysis: Find where Softmax wins ===
df_avg = df_all.groupby(["label", "method"])["pearson"].mean().reset_index()
df_pivot = df_avg.pivot(index="label", columns="method", values="pearson")

others = [m for m in method_dirs.keys() if m != "Softmax"]
df_pivot['best_other'] = df_pivot[others].max(axis=1)
df_pivot['softmax_margin'] = df_pivot['Softmax'] - df_pivot['best_other']

# Get top 10 odors where Softmax is the highest performer
top_10_softmax = (
    df_pivot[df_pivot['softmax_margin'] > 0]
    .sort_values("softmax_margin", ascending=False)
    .head(5)
    .index.tolist()
)

# === Plotting ===
if not top_10_softmax:
    print("No odors found where Softmax is the top performer.")
else:
    df_plot = df_all[df_all["label"].isin(top_10_softmax)].copy()

    # Increase figure size slightly to accommodate larger fonts
    plt.figure(figsize=(14, 10))
    
    ax = sns.boxplot(
        data=df_plot,
        y="label",
        x="pearson",
        hue="method",
        order=top_10_softmax,
        palette="Set2",
        orient="h",
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"}
    )

    # Explicitly setting font sizes to 14 for all text components
    plt.title("Top 5 Odor Descriptors for Softmax", fontsize=18)
    plt.xlabel("Pearson Correlation (r)", fontsize=18)
    plt.ylabel("Odor Descriptor", fontsize=18)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=18, title_fontsize=18)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("top_10_softmax_best.png", dpi=300)
    print("✅ Saved: top_10_softmax_best_font14.png with font size 14")
