import pandas as pd

# 文件路径和权重
file1 = 'predictions_sub2.csv'
file2 = 'predictions_nonparr_catboost.csv'
file3 = 'predictions_ori.csv'
file4 = 'predictions_HLnomodred.csv'
file5 = "predictions_nonparralel_HL.csv"
weight1 = 0.3
weight2 = 0.175
weight3 = 0.175
weight4 = 0.175
weight5 = 0.175

# 读取并设置索引
df1 = pd.read_csv(file1).set_index('stimulus')
df2 = pd.read_csv(file2).set_index('stimulus')
df3 = pd.read_csv(file3).set_index('stimulus')
df4 = pd.read_csv(file4).set_index('stimulus')
df5 = pd.read_csv(file5).set_index('stimulus')

# 取交集并排序
common_index = df1.index.intersection(df2.index).intersection(df3.index).intersection(df4.index).intersection(df5.index)
common_columns = df1.columns.intersection(df2.columns).intersection(df3.columns).intersection(df4.columns).intersection(df5.columns)
all_columns = df1.columns.union(df2.columns).union(df3.columns).union(df4.columns).union(df5.columns)
non_common_columns = all_columns.difference(common_columns)

print("\n❌ 不在所有 DataFrame 中的 column 及其缺失来源：")
for col in non_common_columns:
    print(col)
dataframes = [df1, df2, df3, df4, df5]
df_names = ['df1', 'df2', 'df3', 'df4', 'df5']

# 所有 index 的并集
all_indices = df1.index.union(df2.index).union(df3.index).union(df4.index).union(df5.index)
non_common_indices = all_indices.difference(common_index)

print("❌ 不在所有 DataFrame 中的 index 及其缺失来源：")
for idx in non_common_indices:
    missing_in = [name for df, name in zip(dataframes, df_names) if idx not in df.index]
    print(f"Index {idx} 缺失于: {missing_in}")


df1_common = df1.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df2_common = df2.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df3_common = df3.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df4_common = df4.loc[common_index, common_columns].sort_index().sort_index(axis=1)
df5_common = df5.loc[common_index, common_columns].sort_index().sort_index(axis=1)

# 加权平均
df_weighted = df1_common * weight1 + df2_common * weight2 + df3_common * weight3 + df4_common*weight4 +df5_common*weight5

# 恢复 stimulus 列并保存
df_weighted.reset_index().to_csv('predictions.csv', index=False)

