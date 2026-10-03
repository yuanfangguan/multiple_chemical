import pandas as pd
import numpy as np

# ======================================================
# 1. Load predicted fingerprint file
# ======================================================
fp = pd.read_csv("predicted_full_fingerprints.csv")
fp_values = fp.values  # numpy array

print("Loaded predicted FP shape:", fp_values.shape)

# ======================================================
# 2. INTEGER COUNT VERSIONS (recommended for enumeration)
# ======================================================

# ---- (A) Simple rounding ------------------------------
fp_round = np.round(fp_values).astype(int)
pd.DataFrame(fp_round, columns=fp.columns).to_csv("pred_fp_round.csv", index=False)

# ---- (B) Round & clip to 0–6 (safe range) -------------
fp_round_clip6 = np.clip(np.round(fp_values), 0, 6).astype(int)
pd.DataFrame(fp_round_clip6, columns=fp.columns).to_csv("pred_fp_round_clip6.csv", index=False)

# ---- (C) Scaled ×3 then rounded -----------------------
#     Makes counts less sparse
fp_scaled3 = np.clip(np.round(fp_values * 3), 0, 6).astype(int)
pd.DataFrame(fp_scaled3, columns=fp.columns).to_csv("pred_fp_scaled3.csv", index=False)

# ---- (D) Scaled ×6 then rounded -----------------------
#     Produces even richer count fingerprints
fp_scaled6 = np.clip(np.round(fp_values * 6), 0, 6).astype(int)
pd.DataFrame(fp_scaled6, columns=fp.columns).to_csv("pred_fp_scaled6.csv", index=False)

print("Saved integer-count fingerprint test files.")

# ======================================================
# 3. (OPTIONAL) your original binary thresholds
# ======================================================

fp_bin_0_5 = (fp_values > 0.05).astype(int)
pd.DataFrame(fp_bin_0_5, columns=fp.columns).to_csv("pred_fp_binary_0.05.csv", index=False)

fp_bin_0_4 = (fp_values > 0.02).astype(int)
pd.DataFrame(fp_bin_0_4, columns=fp.columns).to_csv("pred_fp_binary_0.02.csv", index=False)

fp_bin_0_2 = (fp_values > 0.1).astype(int)
pd.DataFrame(fp_bin_0_2, columns=fp.columns).to_csv("pred_fp_binary_0.1.csv", index=False)

print("Also saved old binary versions (for comparison).")
print("Done.")

