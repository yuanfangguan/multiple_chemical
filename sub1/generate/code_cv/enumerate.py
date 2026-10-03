import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# ================================================================
# 1. Load predicted fingerprints
# ================================================================
pred = pd.read_csv("predicted_full_fingerprints.csv")
pred = pred.values.astype(float)
print("Predicted fingerprints:", pred.shape)

# ================================================================
# 2. Load real fingerprints (created by your molsig preprocessing)
# ================================================================
real_df = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")

# Extract ECFP bits only
bit_cols = [c for c in real_df.columns if c.startswith("bit_")]
real_fp = real_df[bit_cols].values.astype(float)

print("Real fingerprint DB:", real_fp.shape)

# ================================================================
# 3. Load test.csv (smell test set)
# ================================================================
test = pd.read_csv("test.csv")
test_stimuli = test["stimulus"].unique()
print("Test-set stimuli:", len(test_stimuli))

# ================================================================
# 4. Load stimulus → molecule mapping
# ================================================================
stim_map = pd.read_csv("../../data/raw/TASK1_Stimulus_definition.csv")
# Columns: stimulus, molecule, ...

# Keep only mappings for stimuli in the test set
test_map = stim_map[stim_map["stimulus"].isin(test_stimuli)]

# molecule column = actual CID (string)
test_cids = set(test_map["molecule"].astype(str))
print("Test-set molecule IDs:", len(test_cids))

# ================================================================
# 5. Restrict real fingerprints to only test-set molecules
# ================================================================
mask = real_df["molecule"].astype(str).isin(test_cids)
real_fp_test = real_fp[mask]
meta_test = real_df.loc[mask, ["molecule", "SMILES"]].reset_index(drop=True)

print("Real fingerprints restricted to test set:", real_fp_test.shape)

# If nothing matched → fail early
if real_fp_test.shape[0] == 0:
    raise ValueError("No overlap between test-set molecules and fingerprint DB.")

# ================================================================
# 6. Compute cosine similarity
#    (cosine sim = 1 – cosine distance)
# ================================================================
cosine_sim = 1 - cdist(pred, real_fp_test, metric='cosine')

# ================================================================
# 7. For each predicted fingerprint → find nearest real molecule
# ================================================================
results = []

for i in range(pred.shape[0]):
    sims = cosine_sim[i]
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]

    mol = meta_test.iloc[best_idx]["molecule"]
    smi = meta_test.iloc[best_idx]["SMILES"]

    print(f"\n=============================")
    print(f"Pred FP {i}")
    print(f"Closest test molecule: CID {mol}")
    print(f"Similarity: {best_sim:.4f}")
    print(f"SMILES: {smi}")

    results.append({
        "pred_index": i,
        "best_cid": mol,
        "best_smiles": smi,
        "similarity": best_sim
    })

# ================================================================
# 8. Save results
# ================================================================
out = pd.DataFrame(results)
out.to_csv("nearest_test_molecule.csv", index=False)

print("\nSaved nearest_test_molecule.csv")

