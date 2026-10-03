import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Define strategy directories ===
method_dirs = {
    "Mean": "../code_paper_xgboost/new_nonparrallel_noRATA_meanonly/",
    "Max": "../code_paper_xgboost/new_nonparrallel_noRATA_maxonly/",
    "Mean+Max": "../code_paper_xgboost/new_nonparrallel_noRATA/",
    "Softmax": "../code_paper_A_xgboost/new_nonparrallel_noRATA/",
    "Gated": "../code_paper_B_xgboost/new_nonparrallel_noRATA/",
    # "Nonlinear": "../code_paper_C/new_nonparrallel_noRATA/",
    "Single": "../code_paper_xgboost/newbase_sep_catboost_noRATA/"
}

records = []

# === Extract only MEAN row from each evaluation.tsv.[0-4] ===
for method, path in method_dirs.items():
    for i in range(5):
        eval_file = os.path.join(path, f"evaluation.tsv.{i}")
        if not os.path.exists(eval_file):
            print(f"⚠️ Missing: {eval_file}")
            continue

        df = pd.read_csv(eval_file, sep="\t")
        mean_row = df[df["label"] == "MEAN"].copy()
        if not mean_row.empty:
            mean_row["fold"] = i
            mean_row["method"] = method
            records.append(mean_row[["method", "fold", "pearson", "cosine"]])

df_means = pd.concat(records, ignore_index=True)
print(df_means)

# === Plot Pearson ===
plt.figure(figsize=(8,6))
sns.boxplot(data=df_means, x="method", y="pearson", palette="Set2")
sns.swarmplot(data=df_means, x="method", y="pearson", color="black", size=6, alpha=0.6)

plt.title("Average Predictive Performance (XGBoost)", fontsize=18)
plt.ylabel("Pearson Correlation (Mean per Fold)", fontsize=18)
plt.xlabel("", fontsize=18)
plt.xticks(fontsize=18, rotation=15, ha="right")
plt.yticks(fontsize=18)

plt.tight_layout()
plt.savefig("mean_only_pooling_pearson.png", dpi=300)
plt.close()

# === Plot Cosine ===
plt.figure(figsize=(8,6))
sns.boxplot(data=df_means, x="method", y="cosine", palette="Set2")
sns.swarmplot(data=df_means, x="method", y="cosine", color="black", size=6, alpha=0.6)

plt.title("Average Predictive Performance (XGBoost)", fontsize=18)
plt.ylabel("Cosine Distance (Mean per Fold)", fontsize=18)
plt.xlabel("", fontsize=18)
plt.xticks(fontsize=18, rotation=15, ha="right")
plt.yticks(fontsize=18)

plt.tight_layout()
plt.savefig("mean_only_pooling_cosine.png", dpi=300)
plt.close()

print("✅ Saved figures: mean_only_pooling_pearson.png and mean_only_pooling_cosine.png")

