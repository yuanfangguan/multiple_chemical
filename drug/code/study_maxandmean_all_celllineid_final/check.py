import pandas as pd
import os

# ========================
# --- CONFIGURATION ---
# ========================
train_file = "train.csv"
validation_file = "../../data/triplet_external_validation.csv"
drug_file = "../../data/drug.csv"

def normalize_cid(x):
    try: return str(int(float(x)))
    except: return str(x)

# ========================
# --- 1. PREPARE MAPS ---
# ========================
drug_df = pd.read_csv(drug_file, dtype=str)
drug_map = {str(name).lower(): normalize_cid(cid) for name, cid in zip(drug_df["dname"], drug_df["cid"])}

# ========================
# --- 2. BUILD REGISTRY ---
# ========================
print("🔍 Indexing training pairs...")
# Added low_memory=False to resolve the DtypeWarning
train_df = pd.read_csv(train_file, low_memory=False)
train_df['cid_row'] = train_df['drug_row'].map(drug_map)
train_df['cid_col'] = train_df['drug_col'].map(drug_map)

# Create a set of frozensets (order-independent: {A,B} == {B,A})
train_pair_registry = set()
for _, row in train_df.dropna(subset=['cid_row', 'cid_col']).iterrows():
    train_pair_registry.add(frozenset([row['cid_row'], row['cid_col']]))

# ========================
# --- 3. CHECK TRIPLETS ---
# ========================
print("\n--- Zero-Shot Exposure Audit ---")
triplets_df = pd.read_csv(validation_file, names=["d1", "d2", "d3"])

exposure_summary = []

for _, row in triplets_df.iterrows():
    d_names = [str(row["d1"]).lower(), str(row["d2"]).lower(), str(row["d3"]).lower()]
    cids = [drug_map.get(n) for n in d_names]
    
    if None in cids:
        missing = [d_names[i] for i, c in enumerate(cids) if c is None]
        print(f"⚠️  Skipping {d_names}: Missing CIDs for {missing}")
        continue

    # Define the 3 possible pairs in this triplet
    # Each item: (DrugName1, DrugName2, FrozenSetOfCIDs)
    triplet_pairs = [
        (d_names[0], d_names[1], frozenset([cids[0], cids[1]])),
        (d_names[1], d_names[2], frozenset([cids[1], cids[2]])),
        (d_names[0], d_names[2], frozenset([cids[0], cids[2]]))
    ]
    
    # Corrected list comprehension logic
    seen = []
    for d_n1, d_n2, p_set in triplet_pairs:
        if p_set in train_pair_registry:
            seen.append(f"{d_n1}+{d_n2}")
    
    print(f"🧪 Triplet: {' + '.join(d_names)}")
    print(f"   Exposure: {len(seen)}/3 pairs seen in training.")
    if seen:
        print(f"   Seen Pairs: {', '.join(seen)}")
    else:
        print(f"   🌟 PURE ZERO-SHOT: No constituent pairs were seen in training.")
    print("-" * 30)

    exposure_summary.append({
        "Triplet": " + ".join(d_names),
        "Pairs_Seen_Count": len(seen),
        "Seen_Pairs_List": seen
    })

# pd.DataFrame(exposure_summary).to_csv("triplet_exposure_audit.csv", index=False)
