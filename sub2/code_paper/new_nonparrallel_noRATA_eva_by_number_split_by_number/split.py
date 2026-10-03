import pandas as pd
import os

# === File paths ===
REF_PATH = "../../data/raw/TASK2_Train_mixture_Dataset.csv"
CID_MAP_PATH = "../../data/raw/TASK2_Component_definition.csv"
STIM_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"

OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load datasets ===
df_ref = pd.read_csv(REF_PATH)
df_comp_map = pd.read_csv(CID_MAP_PATH)
df_stim_map = pd.read_csv(STIM_MAP_PATH)

# === Build component → CID mapping ===
component_to_cid = dict(zip(df_comp_map['id'], df_comp_map['CID']))
component_to_dilution = dict(zip(df_comp_map['id'], df_comp_map['dilution']))

# === Compute number of components per stimulus ===
def get_num_components(components_str):
    comps = str(components_str).split(';')
    valid = [c for c in comps if c.strip().isdigit()]
    return len(valid)

df_stim_map['num_components'] = df_stim_map['components'].apply(get_num_components)

# === Merge num_components info into the reference dataset ===
df_ref = df_ref.merge(df_stim_map[['id', 'num_components']], left_on='stimulus', right_on='id', how='left')
df_ref = df_ref.drop(columns=['id'])

# === Find unique groups ===
unique_groups = sorted(df_ref['num_components'].dropna().unique())
print(f"Found groups: {unique_groups}")

# === Split per group ===
for n in unique_groups:
    df_test = df_ref[df_ref['num_components'] == n]
    df_train = df_ref[df_ref['num_components'] != n]

    test_path = os.path.join(OUTPUT_DIR, f"test_numcomp_{n}.csv")
    train_path = os.path.join(OUTPUT_DIR, f"train_numcomp_{n}.csv")

    df_train.drop(columns=['num_components']).to_csv(train_path, index=False)
    df_test.drop(columns=['num_components']).to_csv(test_path, index=False)

    print(f"✅ Generated: train_numcomp_{n}.csv ({len(df_train)} rows) + test_numcomp_{n}.csv ({len(df_test)} rows)")

