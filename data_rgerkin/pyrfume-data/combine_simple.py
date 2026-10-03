import os
import pandas as pd

# 获取当前目录下所有子目录
all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

combined_rows = []

for d in all_dirs:
    file_path = os.path.join(d, 'molecules.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # 提取 IsomericSMILES
        smiles = df['IsomericSMILES']

        # 提取要拼接的部分，去掉 CID 和 IsomericSMILES
        other_columns = [col for col in df.columns if col not in ['CID', 'IsomericSMILES']]

        # 拼接字符串
        combined_strings = df[other_columns].astype(str).apply(lambda row: ''.join(row.values), axis=1)

        # 收集结果
        for smi, combined in zip(smiles, combined_strings):
            combined_rows.append([smi, combined])

# 转成 DataFrame
result_df = pd.DataFrame(combined_rows, columns=['IsomericSMILES', 'Combined'])

# 保存
result_df.to_csv('simple_combined_molecules.csv', index=False)
print('已保存到 simple_ombined_molecules.csv')

