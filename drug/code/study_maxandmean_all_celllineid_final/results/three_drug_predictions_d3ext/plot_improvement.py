import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# --- 1. 分类映射逻辑 ---
# ========================
def get_cancer_type(cell_name):
    name = str(cell_name).upper()
    if any(x in name for x in ['MCF7', 'MDA-MB', 'BT-', 'HCC', 'T-47D', 'ZR75', 'EFM', 'CAMA', 'EVSA']):
        return 'Breast'
    if any(x in name for x in ['NCI-H', 'CALU', 'A549', 'HOP', 'EKVX']):
        return 'Lung'
    if any(x in name for x in ['COLO', 'HCT', 'SW', 'LS', 'LOVO', 'DLD1', 'RKO']):
        return 'Colon'
    if any(x in name for x in ['A375', 'A2058', 'SK-MEL', 'MALME', 'LOX', 'WM', 'M14']):
        return 'Melanoma'
    if any(x in name for x in ['OVCAR', 'SK-OV', 'A2780', 'IGROV', 'ES2']):
        return 'Ovarian'
    if any(x in name for x in ['PC-3', 'LNCAP', 'DU-145', 'VCAP']):
        return 'Prostate'
    if any(x in name for x in ['786-0', 'ACHN', 'CAKI', 'SN12C', 'RXF']):
        return 'Renal'
    if any(x in name for x in ['CCRF', 'HL-60', 'K-562', 'MOLT', 'RPMI', 'SR']):
        return 'Leukemia'
    if any(x in name for x in ['SNU', 'AGS', 'KATO']):
        return 'Gastric'
    if any(x in name for x in ['U251', 'SF-', 'SNB']):
        return 'CNS'
    return 'Others'

def draw_clean_violin_with_baseline(input_file):
    df = pd.read_csv(input_file)
    df['Cancer_Type'] = df['cell_line'].apply(get_cancer_type)
    
    # 按中位数降序排列
    order = df.groupby('Cancer_Type')['diff'].median().sort_values(ascending=False).index

    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = 30
    sns.set_style("white") 

    fig, ax = plt.subplots(figsize=(22, 12))

    # --- 新增：在小提琴图下方先画一条红线 ---
    # zorder=1 确保它在最底层，不遮挡小提琴的内部细节
    ax.axhline(0, color='#d62728', linestyle='-', linewidth=3, alpha=0.9, zorder=1)

    # 绘制纯小提琴图
    sns.violinplot(
        x='Cancer_Type', y='diff', data=df, order=order,
        palette="flare",   
        inner='quartile',  
        linewidth=2.5,     
        cut=0,             
        ax=ax,
        zorder=2           # 确保小提琴在红线之上
    )

    # 调整分位数线的样式
    for line in ax.lines:
        if line.get_linestyle() == '--': # 只修改分位数虚线，不影响刚才画的实线红线
            line.set_linewidth(1.5)
            line.set_color('white')
            line.set_alpha(0.8)

    # 细节微调
    plt.xticks(rotation=45, ha='right', fontsize=28)
    plt.yticks(fontsize=28)
    
    plt.xlabel("Cancer Type", fontsize=30, labelpad=20)
    plt.ylabel("Drug sensitivity improvement", fontsize=30, labelpad=20)
    plt.title("Efficacy Gain Distribution by Lineage", fontsize=36,  pad=40)

    # 彻底精简边框
    sns.despine(left=True, bottom=False) 
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.3) 

    plt.tight_layout()
    
    output_name = "Clean_Violin_RedBaseline.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✨ 带有红线基准的小提琴图已生成: {output_name}")
    plt.show()

if __name__ == "__main__":
    draw_clean_violin_with_baseline("exhaustive_css_ri_shard_0.csv")
