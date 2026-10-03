import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Load predicted fingerprints (74 × 2048)
# -------------------------------------------------------------
pred_fp = pd.read_csv("predicted_full_fingerprints.csv")
print("Predicted FP:", pred_fp.shape)

# -------------------------------------------------------------
# Load test.csv → get stimuli
# -------------------------------------------------------------
test = pd.read_csv("test.csv")
stimuli = test["stimulus"].tolist()
print("Test stimuli:", len(stimuli))

# -------------------------------------------------------------
# Map stimuli → CIDs
# -------------------------------------------------------------
stim_map = pd.read_csv("../../data/raw/TASK1_Stimulus_definition.csv")

test_rows = stim_map[stim_map["stimulus"].isin(stimuli)]
test_cids = test_rows["molecule"].astype(str).tolist()

print("Mapped test CIDs:", len(test_cids))

# -------------------------------------------------------------
# Load real fingerprints
# -------------------------------------------------------------
real_fp = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")
real_fp["molecule"] = real_fp["molecule"].astype(str)

# Ensure ordering matches predicted fingerprints (stimulus order)
real_fp_ordered = real_fp.set_index("molecule").loc[test_cids].reset_index()
print("Real FP subset:", real_fp_ordered.shape)

real_bits = real_fp_ordered[[f"bit_{i}" for i in range(2048)]].values.astype(float)
pred_bits = pred_fp.values.astype(float)

# -------------------------------------------------------------
# Compute correlation matrix
# -------------------------------------------------------------
N = len(pred_bits)
corr = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        v1 = pred_bits[i]
        v2 = real_bits[j]

        if np.std(v1) == 0 or np.std(v2) == 0:
            corr[i, j] = 0
        else:
            corr[i, j] = np.corrcoef(v1, v2)[0, 1]

corr_df = pd.DataFrame(corr, columns=[f"cid_{cid}" for cid in test_cids])
corr_df.to_csv("fp_corr_test_only.csv", index=False)

print("Saved fp_corr_test_only.csv, shape:", corr_df.shape)

