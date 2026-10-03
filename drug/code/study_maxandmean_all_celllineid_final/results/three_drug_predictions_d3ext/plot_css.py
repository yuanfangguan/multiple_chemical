import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# --- 配置与数据加载 ---
# ========================
input_file = "exhaustive_css_ri_shard_0.csv"
output_file = "CSS_Triplet_vs_Pair_Scatter_BigFont.png"

# 设置全局字体为 Arial，并将字号设为 30
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 30  # 全局基础字号

def draw_css_scatter(file_path):
    df = pd.read_csv(file_path)

    # 调大画布尺寸，以容纳 30 号的大字体
    fig, ax = plt.subplots(figsize=(12, 12)) 

    # 绘制散点图
    scatter = ax.scatter(
        df['pair_score'], 
        df['triplet_score'], 
        c=df['diff'],      
        cmap='viridis',    
        s=150,           # 配合大字号，调大散点尺寸
        alpha=0.6, 
        edgecolors='none'
    )

    # 绘制 y=x 对角参考线
    limit = max(df['pair_score'].max(), df['triplet_score'].max())
    min_limit = min(df['pair_score'].min(), df['triplet_score'].min())
    ax.plot([min_limit, limit], [min_limit, limit], color='gray', linestyle='--', linewidth=2, label='y=x')

    # 设置坐标轴标签与标题，显式指定 fontsize=30
    ax.set_xlabel("Pairwise score", fontsize=30, labelpad=15)
    ax.set_ylabel("Max triplet score", fontsize=30, labelpad=15)
    ax.set_title("Combination sensitivity score", fontsize=30,  pad=25)

    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=30)

    # 添加颜色条并设置其字体大小
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Synergy Gain', rotation=270, labelpad=40, fontsize=30)
    cbar.ax.tick_params(labelsize=30) # 颜色条刻度字号

    sns.despine() 
    ax.grid(True, linestyle=':', alpha=0.4) 

    # 自动调整布局，防止大字号超出边界
    plt.tight_layout()
    
    # 保存高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✨ 大字号散点图已生成: {output_file}")
    plt.show()

if __name__ == "__main__":
    draw_css_scatter(input_file)
