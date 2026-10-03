import pandas as pd

# Load files
real_fp = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")
test = pd.read_csv("test.csv")
stim_map = pd.read_csv("../../data/raw/TASK1_Stimulus_definition.csv")

# Convert test stimuli → CID
test_stimuli = set(test["stimulus"])
subset = stim_map[stim_map["stimulus"].isin(test_stimuli)]
test_cids = subset["molecule"].astype(str).tolist()

print("Number of test stimuli:", len(test_stimuli))
print("Number of test CIDs:", len(test_cids))
print("\nExample CIDs from test:")
print(test_cids[:20])

print("\nExample CIDs in real_fp:")
print(real_fp["molecule"].astype(str).head().tolist())

missing = [cid for cid in test_cids if cid not in real_fp["molecule"].astype(str).tolist()]
print("\nMissing CIDs (not found in fingerprint file):", missing)
print("Count missing:", len(missing))

