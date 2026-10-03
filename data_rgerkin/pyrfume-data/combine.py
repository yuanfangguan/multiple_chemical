import os
import pandas as pd

# 获取当前目录下所有子目录
all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

combined_rows = []

for d in all_dirs:
    molecules_file = os.path.join(d, 'molecules.csv')
    print(molecules_file)
    if not os.path.exists(molecules_file):
        continue

    # 读取 molecules.csv
    mol_df = pd.read_csv(molecules_file)
    smiles_list = mol_df['IsomericSMILES']
    cid_list = mol_df['CID']

    # 找到当前目录下所有其他 csv 文件
    other_files = [f for f in os.listdir(d) if f.endswith('.csv') and f != 'molecules.csv']

    # 读取所有其他 csv 文件，合并成一个大表
    other_dfs = []
    for f in other_files:
        print(f)
        file_path = os.path.join(d, f)
        try:
            df = pd.read_csv(file_path, encoding='utf-8', engine='python')
            if 'CID' in df.columns:
                other_dfs.append(df)
        except Exception as e:
            print(f"跳过无法读取的文件: {file_path}. 错误: {e}")

    if other_dfs:
        merged_df = pd.concat(other_dfs, ignore_index=True)
    else:
        merged_df = pd.DataFrame()

    # 遍历 molecules.csv 里的每一行
    for cid, smiles in zip(cid_list, smiles_list):
        combined_string = ''

        if not merged_df.empty:
            matched_rows = merged_df[merged_df['CID'] == cid]
            if not matched_rows.empty:
                for _, row in matched_rows.iterrows():
                    other_columns = [col for col in merged_df.columns if col != 'CID']
                    combined_string += ''.join(str(row[col]) for col in other_columns)

        # 收集结果
        combined_rows.append([smiles, combined_string])

# 转成 DataFrame
result_df = pd.DataFrame(combined_rows, columns=['IsomericSMILES', 'Combined'])

# 保存
result_df.to_csv('combined_molecules.csv', index=False)
print('已保存到 combined_molecules.csv')

