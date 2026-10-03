#!/usr/bin/env python3
import pandas as pd
import sys

# 设定 Seed
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
print(f"Using seed = {seed}")

# 1. 加载数据并提取唯一药物
# 这里直接复用你之前的清洗逻辑
df = pd.read_csv("../../data/TWOSIDES_drugpair_clean_MRF.csv")
all_smiles = pd.concat([df["drug_1_smiles"], df["drug_2_smiles"]]).unique()
print(f"Total unique drugs identified: {len(all_smiles)}")

# 2. 生成单药推理对 (Single Drug Pairs)
# 格式为: drug_1_smiles = SMILES_A, drug_2_smiles = SMILES_A
# 这样模型在推理时会计算该药物自身的“背景频率”
single_list = pd.DataFrame({
    "drug_1_smiles": all_smiles,
    "drug_2_smiles": all_smiles
})

# 3. 添加 dummy target 保持格式一致
single_list["mean_reporting_frequency"] = 0.0

# 4. 保存文件
output_name = "test_single_drugs.csv"
single_list.to_csv(output_name, index=False)

print(f"✔ Saved {output_name} ({len(single_list)} drugs)")

# 展示前几行示例
print("\nSample of single drug list:")
print(single_list.head())
