import pandas as pd
import os
import joblib
import numpy as np
import random

# === File paths ===
TEST_PATH = "test.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
COMPONENT_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
CID_SMILES_PATH = "../../data/raw/CID.csv"

FEATURE_SET_PATHS = {
    "maccs": "../../data/processed/features_maccs.csv",
    "morgan": "../../data/processed/features_morgan.csv",
    "rdkitfp": "../../data/processed/features_rdkitfp.csv",
    "descriptors": "../../data/processed/features_descriptors.csv",
    "mordred": "../../data/raw/Mordred_Descriptors.csv"
}

MODEL_NAME = "combined_no_rata"
# 1. 修改为训练代码实际保存的路径 concat_pool
MODEL_DIR = f"models/{MODEL_NAME}/concat_pool"   
OUTPUT_CSV = "predictions.csv"

# === Load data ===
df_test = pd.read_csv(TEST_PATH)
df_stim_map = pd.read_csv(STIMULUS_MAP_PATH)
df_comp_map = pd.read_csv(COMPONENT_MAP_PATH)
df_cid = pd.read_csv(CID_SMILES_PATH)

df_test = df_test[~df_test["stimulus"].isin(["AN873"])]

# === Build mappings ===
component_to_cid = dict(zip(df_comp_map["id"], df_comp_map["CID"]))
component_to_dilution = dict(zip(df_comp_map["id"], df_comp_map["dilution"]))

# === Merge chemical feature sets ===
merged_feats = {}
for name, path in FEATURE_SET_PATHS.items():
    print(f"Loading chemical feature set: {name}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    if "SMILES" in df.columns:
        df = df.drop(columns=["SMILES"])

    df = df.set_index("molecule")
    for cid, row in df.iterrows():
        merged_feats.setdefault(cid, []).extend(row.values.tolist())

df_chem_feats = pd.DataFrame.from_dict(merged_feats, orient="index")
df_chem_feats.index.name = "molecule"
df_chem_feats = df_chem_feats.fillna(0)

print(f"✅ Loaded combined chemical feature matrix: {df_chem_feats.shape}")

# === Build Stimulus → component pairs ===
def get_pairs(components_str):
    comps = str(components_str).split(";")
    out = []
    for s in comps:
        if s.strip().isdigit():
            comp = int(s)
            cid = component_to_cid.get(comp)
            dil = component_to_dilution.get(comp)
            if cid in df_chem_feats.index and dil is not None:
                out.append((cid, float(dil)))
    return out

stim2pairs = {}
for _, r in df_stim_map[df_stim_map["id"].isin(df_test["stimulus"])].iterrows():
    pairs = get_pairs(r["components"])
    if pairs:
        stim2pairs[r["id"]] = pairs

# === 2. 核心修改：对齐训练集的 Concatenation + Oversampling 逻辑 ===
fp_dim = df_chem_feats.shape[1]
MAX_COMPONENTS = 10

X_list = []
stim_ids = []
num_comp_list = [] 

# 设置相同的随机种子，保证过采样行为与训练集一致（如果测试集也有多于/少于的情况）
random.seed(42)

for _, row in df_test.iterrows():
    stim = row["stimulus"]
    if stim not in stim2pairs:
        print(f"⚠️ No features for stimulus {stim}")
        continue

    pairs = stim2pairs[stim]
    n_chems = len(pairs) # 记录原本真实的化学品数量

    if n_chems == 0:
        # 空混合物回退机制
        feat = np.zeros(MAX_COMPONENTS * (fp_dim + 1))
    else:
        component_vectors = []
        for cid, d in pairs:
            v = df_chem_feats.loc[cid].values.astype(float)
            v_with_dil = np.append(v, float(d))
            component_vectors.append(v_with_dil)
        
        # --- 随机超采样逻辑 (无论测试集原本是3个还是5个，都采样撑满到10个) ---
        while len(component_vectors) < MAX_COMPONENTS:
            sampled_comp = random.choice(component_vectors)
            component_vectors.append(sampled_comp)
        
        # 如果超过 10 个则截断
        if len(component_vectors) > MAX_COMPONENTS:
            component_vectors = component_vectors[:MAX_COMPONENTS]
            
        # 展平成一维长向量
        feat = np.concatenate(component_vectors)

    X_list.append(feat)
    stim_ids.append(stim)
    num_comp_list.append(n_chems)    # 保存原始化学品数量 (3, 5, 或 10)

# === 3. 生成与训练集结构完全一致的列名 ===
cols = []
for c_idx in range(MAX_COMPONENTS):
    cols.extend([f"comp_{c_idx}_fp_{i}" for i in range(fp_dim)])
    cols.append(f"comp_{c_idx}_dilution")

X_all = pd.DataFrame(np.array(X_list), columns=cols)
# 确保数据类型为数值且填充空值
X_all = X_all.apply(pd.to_numeric, errors='coerce').fillna(0)

# === Load models and predict ===
final_preds = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.endswith(".pkl"):
        label = fname.replace("model_","").replace(".pkl","")
        model = joblib.load(os.path.join(MODEL_DIR, fname))
        print(f"Predicting for label: {label}")
        final_preds[label] = model.predict(X_all)

# === Build output ===
df_out = pd.DataFrame({"stimulus": stim_ids})
for label in sorted(final_preds):
    df_out[label] = final_preds[label]

# 添加你需要的 num_components 列
df_out["num_components"] = num_comp_list   

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n🎉 Saved predictions with num_components to {OUTPUT_CSV}")
