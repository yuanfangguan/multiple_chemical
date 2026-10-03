import pandas as pd

df = pd.read_csv("all_cell_triplet_predictions.csv")

# 专门针对图中看到的 2500+ 异常值进行捕获
outliers = df[df['synergy_zip'] > 500]

print("🚨 发现以下异常行：")
print(outliers[['Triplet', 'Cell_Line', 'synergy_zip']])

# 统计这些异常值对应的原始 Cancer Type 映射
# 看看是不是只有特定的一两个细胞系（比如 COLO 205 等）出问题
