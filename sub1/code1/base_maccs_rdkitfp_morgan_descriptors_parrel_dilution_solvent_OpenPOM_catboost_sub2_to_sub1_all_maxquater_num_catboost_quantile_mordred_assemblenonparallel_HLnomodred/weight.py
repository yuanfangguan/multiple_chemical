import pandas as pd

# 文件路径和权重
file1 = 'predictions_sub2.csv'
file2 = 'predictions_nonparr_catboost.csv'
file3 = 'predictions_ori.csv'
file4 = 'predictions_HLnomodred.csv'
weight1 = 0.3
weight2 = 0.23
weight3 = 0.23
weight4 = 0.24

# 读取并设置索引
df1 = pd.read_csv(file1).set_index('stimulus')
df2 = pd.read_csv(file2).set_index('stimulus')
df3 = pd.read_csv(file3).set_index('stimulus')
df4 = pd.read_csv(file4).set_index('stimulus')

# 取交集并排序
common_index = df1.index.intersection(df2.index).intersection(df3.index).intersection(df4.index)
common_columns = df1.columns.intersection(df2.columns).intersection(df3.columns).intersection(df4.columns)

df1_common = df1.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df2_common = df2.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df3_common = df3.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df4_common = df4.loc[common_index, common_columns].sort_index().sort_index(axis=1)

# 加权平均
df_weighted = df1_common * weight1 + df2_common * weight2 + df3_common * weight3 + df4_common*weight4

# 恢复 stimulus 列并保存
df_weighted.reset_index().to_csv('predictions.csv', index=False)

