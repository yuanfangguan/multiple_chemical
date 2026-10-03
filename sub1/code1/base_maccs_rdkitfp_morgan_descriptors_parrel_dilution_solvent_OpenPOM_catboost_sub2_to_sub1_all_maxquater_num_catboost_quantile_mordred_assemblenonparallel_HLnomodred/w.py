import pandas as pd
import numpy as np

# 文件路径和权重
file1 = 'predictions_sub2.csv'
file2 = 'predictions_nonparr_catboost.csv'
file3 = 'predictions_ori.csv'
weight1 = 0.2
weight2 = 0.4
weight3 = 0.4

# 读取并设置索引
df1 = pd.read_csv(file1).set_index('stimulus')
df2 = pd.read_csv(file2).set_index('stimulus')
df3 = pd.read_csv(file3).set_index('stimulus')

# 取交集并排序
common_index = df1.index.intersection(df2.index).intersection(df3.index)
common_columns = df1.columns.intersection(df2.columns).intersection(df3.columns)

df1_common = df1.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df2_common = df2.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df3_common = df3.loc[common_index, common_columns].sort_index().sort_index(axis=1)

# 加权平均
df_weighted = df1_common * weight1 + df2_common * weight2 + df3_common * weight3

# Quantile normalization (参考第三个文件)
df_weighted_sorted = df_weighted.copy()

for col in df_weighted.columns:
    weighted_vals = df_weighted[col].values
    reference_vals = df3_common[col].values

    # 排序并映射分位数
    sorted_weighted_idx = np.argsort(weighted_vals)
    sorted_reference_vals = np.sort(reference_vals)

    weighted_vals[sorted_weighted_idx] = sorted_reference_vals
    df_weighted_sorted[col] = weighted_vals

# 恢复 stimulus 列并保存
df_weighted_sorted = df_weighted_sorted.reset_index()
df_weighted_sorted.to_csv('predictions_quantile.csv', index=False)

