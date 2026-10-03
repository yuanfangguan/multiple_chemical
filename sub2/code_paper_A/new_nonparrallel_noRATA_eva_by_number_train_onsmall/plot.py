import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === 配置 ===
# 对应你要分析的组件数量和对应的文件名
TARGET_COMPS = [2, 3, 5, 10]
# 你的 5 个评价文件夹列表
GROUP_DIRS = [
    "evaluation_groups_0",
    "evaluation_groups_1",
    "evaluation_groups_2",
    "evaluation_groups_3",
    "evaluation_groups_4",
]
OUTPUT_IMAGE = "pearson_boxplot.png"

# 设置字体大小，使其更易读
plt.rcParams.update({"font.size": 14})

# === Step 1: 收集并聚合所有数据 ===
all_data = []

for group_dir in GROUP_DIRS:
    if not os.path.exists(group_dir):
        print(f"⚠️ 警告: 找不到文件夹 {group_dir}，跳过。")
        continue

    for comp in TARGET_COMPS:
        filename = f"evaluation_numcomp_{comp}.tsv"
        filepath = os.path.join(group_dir, filename)

        if os.path.exists(filepath):
            # 读取 tsv 文件
            df = pd.read_csv(filepath, sep="\t")

            # 过滤掉全局统计行 ALL 和 MEAN，只保留具体的香气标签
            df_filtered = df[~df["label"].isin(["ALL", "MEAN"])].copy()

            # 添加元数据列，方便后续按类别画图
            df_filtered["num_components"] = f"{comp} Components"
            df_filtered["group"] = group_dir

            all_data.append(df_filtered[["num_components", "pearson", "group"]])
        else:
            print(f"⚠️ 提示: {filepath} 不存在。")

if not all_data:
    print("❌ 错误: 未读取到任何有效的评估数据，请检查文件路径。")
    exit()

# 拼接成一个统一的 DataFrame
df_master = pd.concat(all_data, ignore_index=True)

# === Step 2: 画 Box Plot ===
plt.figure(figsize=(10, 6))

# 使用 seaborn 画出漂亮的箱线图
# 如果你想看每个 group 的分布，可以加上 hue="group"；如果想混合在一起只看化学品数量的影响，保持现状即可
sns.boxplot(
    x="num_components",
    y="pearson",
    data=df_master,
    palette="Set2",
    width=0.5,
    linewidth=2,
    showmeans=True,  # 在箱线图中显示均值点
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 8,
    },
)

# === Step 3: 图表美化 ===
plt.title(
    "Pearson Correlation Distribution by Number of Components",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
plt.xlabel("Mixture Complexity", fontsize=14, labelpad=10)
plt.ylabel("Pearson Correlation ($r$)", fontsize=14, labelpad=10)
plt.ylim(-0.1, 1.1)  # 皮尔逊相关系数的理论上限是1.0，根据你的数据调整下限
plt.grid(axis="y", linestyle="--", alpha=0.7)

# === Step 4: 保存与展示 ===
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.show()

print(f"🎉 成功将 Pearson 箱线图保存至: {OUTPUT_IMAGE}")
