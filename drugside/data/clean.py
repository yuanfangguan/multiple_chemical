#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv("TWOSIDES_drugpair_unique.csv")

print("Before cleaning:", len(df))

# 1. Drop rows where PRR is not numeric
df["PRR"] = pd.to_numeric(df["PRR"], errors="coerce")
df = df[df["PRR"].notna()]

# 2. Drop rows with missing SMILES
df = df[df["drug_1_smiles"].notna() & df["drug_2_smiles"].notna()]
df = df[df["drug_1_smiles"] != ""]
df = df[df["drug_2_smiles"] != ""]

# 3. Drop obviously malformed rows (like only PRR present)
df = df[df["drug_1_smiles"].str.len() > 2]
df = df[df["drug_2_smiles"].str.len() > 2]

df["row_id"] = df.index

print("After cleaning:", len(df))

df.to_csv("TWOSIDES_drugpair_clean.csv", index=False)
print("✔ Saved TWOSIDES_drugpair_clean.csv")

