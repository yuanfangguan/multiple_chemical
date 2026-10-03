import pandas as pd
import numpy as np

# ======================================================
# Load predicted fingerprints (N × 2048)
# ======================================================
fp = pd.read_csv("predicted_full_fingerprints.csv")  # each row = 1 predicted molecule

# ======================================================
# Compute bit counts for thresholds
# ======================================================
stats_df = pd.DataFrame()
stats_df["bits_gt_0_5"] = (fp > 0.5).sum(axis=1)
stats_df["bits_gt_0_4"] = (fp > 0.4).sum(axis=1)
stats_df["bits_gt_0_2"] = (fp > 0.2).sum(axis=1)

# Add sample index for readability
stats_df["sample_index"] = np.arange(len(stats_df))

# Reorder columns
stats_df = stats_df[["sample_index", "bits_gt_0_5", "bits_gt_0_4", "bits_gt_0_2"]]

# ======================================================
# Save results
# ======================================================
stats_df.to_csv("predicted_fp_bit_statistics.csv", index=False)

print("Saved statistics → predicted_fp_bit_statistics.csv")
print(stats_df.head())

