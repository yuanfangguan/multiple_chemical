import pandas as pd
import numpy as np
import sys

# 文件路径和权重
file1 = sys.argv[1]
file2 = sys.argv[2]
weight1 = 0.6
weight2 = 0.4

df1 = pd.read_csv(file1).set_index('stimulus')
df2 = pd.read_csv(file2).set_index('stimulus')

common_index = df1.index.intersection(df2.index)
common_columns = df1.columns.intersection(df2.columns)

df1_common = df1.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df2_common = df2.loc[common_index, common_columns].sort_index().sort_index(axis=1)

df_weighted = df1_common * weight1 + df2_common * weight2

# Quantile normalization
df_weighted_sorted = df_weighted.copy()

for col in df_weighted.columns:
    weighted_vals = df_weighted[col].values
    reference_vals = df2_common[col].values

    # Sort both arrays
    sorted_weighted_idx = np.argsort(weighted_vals)
    sorted_reference_vals = np.sort(reference_vals)

    # Apply quantile normalization
    weighted_vals[sorted_weighted_idx] = sorted_reference_vals
    df_weighted_sorted[col] = weighted_vals

# 恢复stimulus为列
df_weighted_sorted = df_weighted_sorted.reset_index()

# 保存
df_weighted_sorted.to_csv('predictions.csv', index=False)

