import pandas as pd
import numpy as np
import joblib
import os
from itertools import combinations
from catboost import Pool

# === 1. 配置 ===
STIMULUS_ID = "AK432"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
MODEL_DIR = "models/combined_no_rata/softmax_pool"

FEATURE_SET_PATHS = [
    ("maccs", "../../data/processed/features_maccs.csv"),
    ("morgan", "../../data/processed/features_morgan.csv"),
    ("rdkitfp", "../../data/processed/features_rdkitfp.csv"),
    ("descriptors", "../../data/processed/features_descriptors.csv"),
    ("mordred", "../../data/raw/Mordred_Descriptors.csv")
]

# === 2. 加载元数据 ===
print(f"--- 启动 AK432 深度物理特征分析 ---")
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === 3. 加载特征并映射名称 ===
merged_feats = {}
physical_feature_names = [] 

for name, path in FEATURE_SET_PATHS:
    print(f"正在读取 {name}...")
    try:
        df_feat = pd.read_csv(path, encoding='utf-8')
    except:
        df_feat = pd.read_csv(path, encoding='latin1')

    if 'molecule' in df_feat.columns: df_feat.set_index('molecule', inplace=True)
    if 'SMILES' in df_feat.columns: df_feat = df_feat.drop(columns=['SMILES'])
    
    physical_feature_names.extend([f"{name}_{c}" for c in df_feat.columns])
    for cid, row in df_feat.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

chem_fp_count = len(physical_feature_names) # 记录化学特征的数量
physical_feature_names.extend(["avg_dilution", "num_chems"])
df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient='index').fillna(0)
print(f"✅ 成功映射了 {len(physical_feature_names)} 个物理特征")

# === 4. 提取混合物 ===
stim_row = df_stim_map[df_stim_map['id'] == STIMULUS_ID]
pairs = []
for s in str(stim_row.iloc[0]['components']).split(';'):
    if s.strip().isdigit():
        cid = component_to_cid.get(int(s))
        dil = component_to_dilution.get(int(s))
        if cid in df_chem_feats.index: pairs.append((cid, float(dil)))

# === 5. 生成子集 (修复 tanh 类型和列名) ===
all_subsets = [list(c) for r in range(1, len(pairs) + 1) for c in combinations(pairs, r)]
X_list = []
alpha = 1.0

for subset in all_subsets:
    # 强制 float64 解决 tanh 报错
    vecs = [df_chem_feats.loc[cid].values.astype(np.float64) for cid, d in subset]
    sc = np.array([d for cid, d in subset])
    exp_s = np.exp(alpha * sc)
    w = exp_s / (exp_s.sum() + 1e-9)
    pooled = (w[:, None] * np.tanh(np.vstack(vecs))).sum(axis=0)
    X_list.append(np.concatenate([pooled, [sc.mean(), len(sc)]]))

# 核心修复：根据报错信息精准对齐列名
# 前 chem_fp_count 个叫 softmax_fp_i，最后两个叫原名
model_feature_names = [f"softmax_fp_{i}" for i in range(chem_fp_count)] + ["avg_dilution", "num_chems"]
X_eval = pd.DataFrame(X_list, columns=model_feature_names).fillna(0)

# === 6. 预测与翻译 ===
report = [f"Final Ablation Report for {STIMULUS_ID}\n" + "="*70]
single_indices = [i for i, s in enumerate(all_subsets) if len(s) == 1]
full_mix_idx = len(all_subsets) - 1

for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith('.pkl'):
        label = fname.replace('model_', '').replace('.pkl', '')
        print(f"分析模型: {label}...")
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        
        preds = model.predict(X_eval)
        max_idx = np.argmax(preds[single_indices])
        max_driver_idx = single_indices[max_idx]
        
        eval_pool = Pool(X_eval.iloc[[max_driver_idx, full_mix_idx]])
        raw_shap = model.get_feature_importance(data=eval_pool, type='ShapValues')
        impact = raw_shap[1, :-1] - raw_shap[0, :-1]
        
        report.append(f"\n[ {label.upper() } ]")
        report.append(f"Full Mix: {preds[full_mix_idx]:.3f} | Max Driver: {preds[max_driver_idx]:.3f}")
        
        top_indices = np.argsort(np.abs(impact))[::-1][:10]
        report.append("关键驱动特征 (已翻译为物理名称):")
        for idx in top_indices:
            p_name = physical_feature_names[idx]
            diff = impact[idx]
            report.append(f"  - {p_name:30}: {'⬆️ 增强' if diff > 0 else '⬇️ 抑制'} {abs(diff):.4f}")

with open(f"physical_insight_{STIMULUS_ID}.txt", "w") as f:
    f.write("\n".join(report))
print(f"✅ 完成！结果已保存至 physical_insight_{STIMULUS_ID}.txt")
