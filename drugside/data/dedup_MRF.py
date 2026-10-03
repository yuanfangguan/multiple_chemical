#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("TWOSIDES_with_smiles.csv", usecols=[
    "drug_1_smiles", "drug_2_smiles", "mean_reporting_frequency"
], low_memory=False)

print("Original rows:", len(df))

# Force numeric, coerce bad/concatenated values to NaN
df["mean_reporting_frequency"] = pd.to_numeric(df["mean_reporting_frequency"], errors="coerce")

bad = df["mean_reporting_frequency"].isna().sum()
if bad > 0:
    print(f"Warning: {bad} rows had unparseable MRF values and will be excluded from the mean")

# Aggregate by drug pair, computing the average MRF
df_unique = df.groupby(["drug_1_smiles", "drug_2_smiles"], as_index=False)["mean_reporting_frequency"].mean()
print("Unique drug pairs:", len(df_unique))

df_unique.to_csv("TWOSIDES_drugpair_unique_MRF.csv", index=False)
print("Saved TWOSIDES_drugpair_unique_MRF.csv")
