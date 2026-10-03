#!/usr/bin/env python3
import pandas as pd
import itertools
import sys

seed = int(sys.argv[1])
print(f"Using seed = {seed}")

# 1. Load and clean the known data
df = pd.read_csv("../../data/TWOSIDES_drugpair_clean_MRF.csv")
df["mean_reporting_frequency"] = pd.to_numeric(df["mean_reporting_frequency"], errors="coerce")
df = df.dropna(subset=["mean_reporting_frequency", "drug_1_smiles", "drug_2_smiles"])
df = df[(df["drug_1_smiles"] != "") & (df["drug_2_smiles"] != "")]

# Save the full training set
df.to_csv("train_full.csv", index=False)
print(f"✔ Saved train_full.csv ({len(df)} pairs)")

# 2. Generate the "Missing" pairs for the 2D matrix
print("Identifying unique drugs and generating missing combinations...")
all_smiles = pd.concat([df["drug_1_smiles"], df["drug_2_smiles"]]).unique()

# Generate all possible unique combinations
all_possible = list(itertools.combinations(all_smiles, 2))
missing_pairs = pd.DataFrame(all_possible, columns=["drug_1_smiles", "drug_2_smiles"])

# Create a set of "Drug1_Drug2" strings to filter existing pairs quickly
def make_set_key(s1, s2):
    return "-".join(sorted([str(s1), str(s2)]))

known_set = set(df.apply(lambda x: make_set_key(x['drug_1_smiles'], x['drug_2_smiles']), axis=1))

# Filter: Keep if the pair is NOT in the known_set
is_new = missing_pairs.apply(lambda x: make_set_key(x['drug_1_smiles'], x['drug_2_smiles']), axis=1).apply(lambda k: k not in known_set)
missing_pairs = missing_pairs[is_new].copy()

# Add a dummy target column so your training/inference code finds the column it expects
missing_pairs["mean_reporting_frequency"] = 0.0

# Save the inference file
missing_pairs.to_csv("test_missing_pairs.csv", index=False)
print(f"✔ Saved test_missing_pairs.csv ({len(missing_pairs)} pairs)")
