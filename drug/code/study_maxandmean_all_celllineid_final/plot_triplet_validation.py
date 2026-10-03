import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# ========================
# --- 1. 癌症类型映射逻辑 ---
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

# ========================
# --- 2. 绘图主函数 ---
# ========================
def plot_triplet_synergy(input_file):
    # 读取预测结果
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    df['Cancer_Type'] = df['Cell_Line'].apply(get_cancer_type)
    
    # 设置全局 Nature 风格样式
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.size'] = 30
    sns.set_style("white")

    # 获取所有唯一的三联药组合
    triplets = df['Triplet'].unique()
    print(f"📊 发现 {len(triplets)} 个药物组合，正在生成图表...")

    for triplet in triplets:
        # 提取当前组合的数据
        sub_df = df[df['Triplet'] == triplet].copy()
        
        # --- 修改点：按照字母顺序排序 ---
        order = sorted(sub_df['Cancer_Type'].unique())

        fig, ax = plt.subplots(figsize=(22, 12))

        # 1. 在底层绘制 y=0 红线基准
        ax.axhline(0, color='#d62728', linestyle='-', linewidth=4, alpha=0.9, zorder=1)

        # 2. 绘制小提琴图 (ZIP Synergy)
        sns.violinplot(
            x='Cancer_Type', y='synergy_zip', data=sub_df, order=order,
            palette="flare",   # 暖色调渐变
            inner='quartile',  # 显示分位数线
            linewidth=3,     
            cut=0,             
            ax=ax,
            zorder=2
        )

        # 3. 美化内部虚线
        for line in ax.lines:
            if line.get_linestyle() == '--':
                line.set_linewidth(2.0)
                line.set_color('white')
                line.set_alpha(0.8)

        # 4. 坐标轴与标题
        plt.xticks(rotation=45, ha='right', fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel("Cancer Type", fontsize=30, fontweight='bold', labelpad=20)
        plt.ylabel("ZIP Synergy Score", fontsize=30, fontweight='bold', labelpad=20)
        
        # 格式化标题
        display_name = triplet.upper()
        plt.title(f"Synergy Distribution: {display_name}", fontsize=34, fontweight='bold', pad=40)

        # 5. 精简边框与参考线
        sns.despine(left=True, bottom=False)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.3)

        plt.tight_layout()
        
        # 6. 保存图片 (文件名逻辑保持不变)
        safe_filename = re.sub(r'[^\w\s-]', '', triplet).strip().replace(' ', '_')
        output_path = f"Synergy_Plot_{safe_filename}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() # 关闭画布节省内存
        
        print(f"✅ 已生成: {output_path}")

if __name__ == "__main__":
    plot_triplet_synergy("all_cell_triplet_predictions.csv")
