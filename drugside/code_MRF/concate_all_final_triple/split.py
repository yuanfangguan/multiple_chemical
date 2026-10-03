#!/usr/bin/env python3
import pandas as pd
import sys

# 增加简单的错误检查
if len(sys.argv) < 2:
    seed = 42
else:
    seed = int(sys.argv[1])
print(f"Using seed = {seed}")

# 1. 加载已知数据 (Known Pairs)
df = pd.read_csv("../../data/TWOSIDES_drugpair_clean_MRF.csv")
df["mean_reporting_frequency"] = pd.to_numeric(df["mean_reporting_frequency"], errors="coerce")
df = df.dropna(subset=["mean_reporting_frequency", "drug_1_smiles", "drug_2_smiles"])

# 确保 SMILES 唯一性标识（排序，防止 A-B 和 B-A 被当成两个）
def sort_pair(s1, s2):
    return tuple(sorted([str(s1), str(s2)]))

# 2. 获取药库中所有的唯一药物
all_drugs = pd.concat([df["drug_1_smiles"], df["drug_2_smiles"]]).unique()
print(f"Unique drugs in library: {len(all_drugs)}")

# 3. 以已知 Pair 为 Seed 生成 Triplet 相关的测试对
print("Generating test triplets based on known pairs...")

triplet_test_rows = []

# 遍历每一个已知的 Pair (A, B)
for index, row in df.iterrows():
    d1 = row['drug_1_smiles']
    d2 = row['drug_2_smiles']
    
    # 遍历药库中所有的药物 C
    for d3 in all_drugs:
        if d3 == d1 or d3 == d2:
            continue
        
        # 对于一个三元组 (A, B, C)，如果你想预测它，
        # 在 DDI 模型中通常需要提供新的组合：(A, C) 和 (B, C)
        # 因为 (A, B) 已经是已知的训练数据了。
        triplet_test_rows.append({'drug_1_smiles': d1, 'drug_2_smiles': d3, 'seed_pair': f"{d1}_{d2}"})
        triplet_test_rows.append({'drug_1_smiles': d2, 'drug_2_smiles': d3, 'seed_pair': f"{d1}_{d2}"})

# 4. 转换为 DataFrame 并去重
# 因为不同的 Seed Pairs 可能会生成相同的测试对 (比如 A-B+C 和 A-C+B)
test_df = pd.DataFrame(triplet_test_rows)

# 移除与训练集重复的对
# 创建 key 用于对比
df['key'] = df.apply(lambda x: sort_pair(x['drug_1_smiles'], x['drug_2_smiles']), axis=1)
test_df['key'] = test_df.apply(lambda x: sort_pair(x['drug_1_smiles'], x['drug_2_smiles']), axis=1)

known_keys = set(df['key'])
test_df = test_df[~test_df['key'].isin(known_keys)]

# 去重并清理
test_df = test_df.drop_duplicates(subset=['key']).drop(columns=['key', 'seed_pair'])

# 5. 保存结果
test_df["mean_reporting_frequency"] = 0.0
test_df.to_csv("test_triplet_based_pairs.csv", index=False)

print(f"✔ Saved test_triplet_based_pairs.csv ({len(test_df)} pairs)")
print(f"This covers all potential new interactions required to evaluate triplets built from known seeds.")
