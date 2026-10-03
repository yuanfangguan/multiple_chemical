import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_dirs = {
    "Descriptors": "../new_descriptors",
    "MACCS": "../new_maccs",
    "Mordred": "../new_mordred",
    "Morgan": "../new_morgan",
    "RDKitFP": "../new_rdkitfp"
}

mean_dirs = {
    "Descriptors": "../new_descriptors_meanonly",
    "MACCS": "../new_maccs_meanonly",
    "Mordred": "../new_mordred_meanonly",
    "Morgan": "../new_morgan_meanonly",
    "RDKitFP": "../new_rdkitfp_meanonly"
}

max_dirs = {
    "Descriptors": "../new_descriptors_maxonly",
    "MACCS": "../new_maccs_maxonly",
    "Mordred": "../new_mordred_maxonly",
    "Morgan": "../new_morgan_maxonly",
    "RDKitFP": "../new_rdkitfp_maxonly"
}

def load_data(folder, model_type):
    dfs = []
    for i in range(5):
        path = os.path.join(folder, f"evaluation.tsv.{i}")
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            df["type"] = model_type
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# 汇总所有数据
all_data = []
for name in base_dirs:
    assemble = load_data(base_dirs[name], "Assemble")
    mean = load_data(mean_dirs[name], "Mean")
    max_ = load_data(max_dirs[name], "Max")
    for df, t in zip([mean, max_, assemble], ["Mean", "Max", "Assemble"]):
        df["feature"] = name
        df["group"] = t
    all_data.append(pd.concat([mean, max_, assemble]))

df_all = pd.concat(all_data)

# 设置字体大小
sns.set_context("talk", font_scale=1.2)

# 自定义颜色（紫、红、蓝）
palette = {"Mean": "#8B5CF6", "Max": "#EF4444", "Assemble": "#3B82F6"}

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_all,
    x="feature",
    y="cosine",       # ← 改成 cosine
    hue="group",
    order=base_dirs.keys(),
    palette=palette,
    linewidth=0.8,
    fliersize=2,
    width=0.6,
    dodge=0.6
)

plt.title("Cosine Distance with LightGBM", fontsize=18)
plt.ylabel("Cosine distance", fontsize=15)
plt.xlabel("Feature group", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(title="Type", fontsize=12, title_fontsize=13, loc="best")

plt.tight_layout()
plt.savefig("feature_group_comparison_boxplot_cosine.png", dpi=300, bbox_inches='tight')
plt.show()

