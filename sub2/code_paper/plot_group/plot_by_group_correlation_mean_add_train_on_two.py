import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 全局字体大小设置
plt.rcParams.update({"font.size": 20})

# === 1. 定义标准 6 种模型的目录 ===
model_dirs = {
    "Mean": "../new_nonparrallel_noRATA_eva_by_number_meanonly/",
    "Max": "../new_nonparrallel_noRATA_eva_by_number_maxonly/",
    "Mean+Max": "../new_nonparrallel_noRATA_eva_by_number/",
    "Softmax": "../../code_paper_A/new_nonparrallel_noRATA_eva_by_number/",
    "Gated": "../../code_paper_B/new_nonparrallel_noRATA_eva_by_num/",
    "Single": "../newbase_sep_catboost_noRATA_eva_by_number/",
    "Subset (44%)": "../../code_paper_A/new_nonparrallel_noRATA_eva_by_number_train_onsmall/",
}

# 新增需要对比的特殊实验路径
TRAIN_BY_TWO_DIR = (
    "../../code_paper_A/new_nonparrallel_noRATA_train_by_number/"
)


def load_eval_data(base_dir, model_label):
    """遍历 fold 和评估文件，计算单次实验的平均皮尔逊相关系数。"""
    results = []
    # 匹配形如 evaluation_groups_0, evaluation_groups_1 等 fold 目录
    fold_dirs = sorted(glob.glob(os.path.join(base_dir, "evaluation_groups_*")))

    for fold_dir in fold_dirs:
        fold_num = os.path.basename(fold_dir).split("_")[-1]
        # 搜索目录下所有的 evaluation_*.tsv 文件
        for file in glob.glob(os.path.join(fold_dir, "evaluation_*.tsv")):
            try:
                df = pd.read_csv(file, sep="\t")
                if "pearson" not in df.columns:
                    continue

                # 优先提取 MEAN 行作为该组（特定化学品数量）的整体表现
                if "label" in df.columns and (df["label"] == "MEAN").any():
                    mean_pearson = df.loc[
                        df["label"] == "MEAN", "pearson"
                    ].values[0]
                else:
                    mean_pearson = df["pearson"].dropna().mean()

                group_name = (
                    os.path.splitext(os.path.basename(file))[0]
                    .replace("evaluation_", "")
                    .replace("numcomp_", "")
                )

                results.append(
                    {
                        "model": model_label,
                        "fold": fold_num,
                        "group": group_name,
                        "pearson": mean_pearson,
                    }
                )
            except Exception as e:
                print(f"Skipping {file}: {e}")
    return pd.DataFrame(results)


# === 2. 加载与处理全部数据 ===
all_data = []

# (A) 加载原本的 6 个基础模型
for label, folder in model_dirs.items():
    if os.path.exists(folder):
        df = load_eval_data(folder, label)
        if not df.empty:
            all_data.append(df)
    else:
        print(f"⚠️ Directory not found: {folder}")

# (B) 加载新提出的 "(train on two comp)" 模型数据
if os.path.exists(TRAIN_BY_TWO_DIR):
    print(f"🔍 Loading special group: (train on two comp)")
    df_special = load_eval_data(TRAIN_BY_TWO_DIR, "train on two comp")
    if not df_special.empty:
        all_data.append(df_special)
else:
    print(f"⚠️ Special directory not found: {TRAIN_BY_TWO_DIR}")

if not all_data:
    print("No data loaded. Please check your directory paths.")
    exit()

df_all = pd.concat(all_data, ignore_index=True)

# === 3. 过滤及排序 (清理 'all') ===
df_all = df_all[df_all["group"] != "all"]


def sort_key(x):
    try:
        return int(x)
    except ValueError:
        return 999


unique_groups = sorted(df_all["group"].unique(), key=sort_key)

# === 4. 绘图 (Plotting) ===
# 既然模型由 6 个变为了 7 个，略微增加高度（10 -> 11）保证箱体不拥挤
plt.figure(figsize=(12, 11))

# Set3 或 Paired 调色板能更好地支持 7 种及以上的离散颜色区分
palette = "Set3"

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
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": "5",
    },
)

# 样式美化
plt.xlabel("Mean Pearson correlation (r)", fontsize=20)
plt.ylabel("Number of chemicals per stimulus", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 给 X 轴设置固定的取值范围 (0 到 0.7)
plt.xlim(0, 0.7)

# 修改图例使其能够容纳 7 个类别（标准6组 + 1组新特定模型）
plt.legend(
    title="Pooling Strategy",
    fontsize=18,  # 图例字体略微调小至 18 以防过宽
    title_fontsize=18,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True,
)

plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()

# 保存图像
output_name = "pearson_by_chemical_number_7groups.png"
plt.savefig(output_name, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Saved: {output_name}")
