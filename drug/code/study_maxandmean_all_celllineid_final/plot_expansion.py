import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

# ========================
# --- 配置路径 ---
# ========================
DATA_FILE = "results/three_drug_predictions_d3ext/exhaustive_synergy_bliss_shard_0.csv" 
FEATURE_FILE = "../../data/features/features_morgan.csv"

plt.rcParams['font.sans-serif'] = "Arial"
sns.set_context("paper", font_scale=1.2)

def normalize_id(x):
    try:
        # 处理 float 转 int 再转 str，确保 "123.0" -> "123"
        return str(int(float(x)))
    except:
        return str(x).strip()

def generate_real_feature_landscape(df):
    # 1. 获取 Top 10 细胞系
    top_cells = df['cell_line'].value_counts().nlargest(10).index.tolist()
    # 过滤掉协同分过低的点，让“山峰”更明显 (可选)
    subset = df[df['cell_line'].isin(top_cells)].copy()

    # 2. 归一化预测结果中的 ID
    subset['added_drug'] = subset['added_drug'].apply(normalize_id)

    # 3. 提取并降维化学特征
    print("🧩 正在加载化学指纹...")
    feat_df = pd.read_csv(FEATURE_FILE)
    feat_df['molecule'] = feat_df['molecule'].apply(normalize_id)

    # 找出在特征文件中存在的药物
    unique_d3 = subset['added_drug'].unique()
    feat_subset = feat_df[feat_df['molecule'].isin(unique_d3)].copy()

    if feat_subset.empty:
        print("❌ 错误：在特征文件中找不到任何匹配的药物 ID！")
        print(f"预测结果样例 ID: {unique_d3[:5]}")
        print(f"特征文件样例 ID: {feat_df['molecule'].iloc[:5].values}")
        return

    print(f"✅ 找到 {len(feat_subset)} 个匹配药物。正在进行 t-SNE 降维...")

    # 准备特征矩阵
    X_feats = feat_subset.drop(columns=['molecule']).values
    # 填充可能存在的 NaN
    X_feats = np.nan_to_num(X_feats)

    # 降维流程
    pca_comp = min(50, X_feats.shape[0], X_feats.shape[1])
    X_pca = PCA(n_components=pca_comp).fit_transform(X_feats)

    # perplexity 不能大于样本数
    perp = min(30, len(feat_subset) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_embedded = tsne.fit_transform(X_pca)

    # 映射回坐标
    coords_map = pd.DataFrame({
        'added_drug': feat_subset['molecule'].values,
        'tsne_1': X_embedded[:, 0],
        'tsne_2': X_embedded[:, 1]
    })

    subset = subset.merge(coords_map, on='added_drug', how='inner')

    # 4. 开始 3D 绘图
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette("husl", 10)

    # 为了防止点太多遮挡，我们可以按协同得分降序排列，让高的点最后画（在最上面）
    subset = subset.sort_values('triplet_score')

    for i, cell in enumerate(top_cells):
        cell_data = subset[subset['cell_line'] == cell]
        ax.scatter(cell_data['tsne_1'], cell_data['tsne_2'], cell_data['triplet_score'],
                   color=colors[i], s=25, label=cell, alpha=0.6, edgecolors='none')

    ax.set_title("Synergy Landscape: Global Expansion Search", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Chemical Similarity (t-SNE 1)", fontsize=12)
    ax.set_ylabel("Chemical Similarity (t-SNE 2)", fontsize=12)
    ax.set_zlabel("Predicted Synergy Score", fontsize=12)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Top 10 Cell Lines")

    plt.tight_layout()
    output_name = "Nature_Figure_4B_Corrected.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✨ 绘图成功：{output_name}")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(DATA_FILE) and os.path.exists(FEATURE_FILE):
        df_all = pd.read_csv(DATA_FILE)
        generate_real_feature_landscape(df_all)
    else:
        print("❌ 错误：请检查路径。")
