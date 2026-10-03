import pandas as pd
import numpy as np

# ======================================================
# 1. Load predicted fingerprints (N × 2048)
# ======================================================
pred_fp = pd.read_csv("predicted_full_fingerprints.csv")

# ======================================================
# 2. Load test stimulus
#    We need stimulus IDs to align molecules
# ======================================================
test_df = pd.read_csv("test.csv")  # must contain: stimulus + smell features

# ======================================================
# 3. Load stimulus → molecule mapping
# ======================================================
stim2mol = pd.read_csv("../../data/raw/TASK1_Stimulus_definition.csv")
stim2mol = stim2mol[["stimulus", "molecule"]]

# attach molecule to test rows
test_df = test_df.merge(stim2mol, on="stimulus", how="left")

# ======================================================
# 4. Load true fingerprint database
# ======================================================
true_fp_all = pd.read_csv("../preprocess/cid_molsig_fp_2048.csv")

# keep only molecule + fingerprint bits
bit_cols = [c for c in true_fp_all.columns if c.startswith("bit_")]
true_fp_all = true_fp_all[["molecule"] + bit_cols]

# ======================================================
# 5. Align true fingerprints to predicted rows via molecule ID
# ======================================================
aligned_true = test_df.merge(true_fp_all, on="molecule", how="left")

# Now extract true fingerprints aligned to prediction rows
true_fp = aligned_true[bit_cols]

print("Aligned true FP shape:", true_fp.shape)
print("Pred FP shape:", pred_fp.shape)

# Sanity check
assert true_fp.shape == pred_fp.shape, "ERROR: shape mismatch after alignment!"

# ======================================================
# 6. Compute correlation bit-by-bit
# ======================================================
correlations = []

for col in bit_cols:
    corr = true_fp[col].corr(pred_fp[col])
    correlations.append(corr)

correlations = np.array(correlations)

print("Bits evaluated:", len(correlations))
print("Mean correlation:", np.nanmean(correlations))
print("Median correlation:", np.nanmedian(correlations))

# ======================================================
# 7. Save detailed results
# ======================================================
corr_df = pd.DataFrame({
    "bit": bit_cols,
    "correlation": correlations
})

corr_df.to_csv("fingerprint_bitwise_correlation.csv", index=False)

print("Saved: fingerprint_bitwise_correlation.csv")

