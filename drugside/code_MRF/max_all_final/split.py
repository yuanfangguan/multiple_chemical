#!/usr/bin/env python3
import pandas as pd
import sys

# We keep the seed for consistency/logging, though not used for KFold anymore
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
print(f"Using seed = {seed} for final model generation")

# Load the full dataset
df = pd.read_csv("../../data/TWOSIDES_drugpair_clean_MRF.csv")
df["row_id"] = df.index

# Save the entire dataset as the final training set
# No splitting logic required
df.to_csv("train_final.csv", index=False)

print(f"✔ Generated final training CSV with {len(df)} rows.")
