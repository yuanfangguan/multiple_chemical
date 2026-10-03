import pandas as pd
import pubchempy as pcp
from functools import lru_cache

# --- Helper: cached lookup for faster repeated queries ---
@lru_cache(None)
def name_to_smiles(name):
    try:
        res = pcp.get_compounds(name, 'name')
        if res:
            return res[0].canonical_smiles
    except:
        return None
    return None

# --- Load file ---
df = pd.read_csv("TWOSIDES.csv")

# --- Convert columns ---
df["drug_1_smiles"] = df["drug_1_concept_name"].apply(name_to_smiles)
df["drug_2_smiles"] = df["drug_2_concept_name"].apply(name_to_smiles)

# --- Remove original name columns OR keep both ---
df = df.drop(columns=["drug_1_concept_name", "drug_2_concept_name"])

# --- Save output ---
df.to_csv("TWOSIDES_with_smiles.csv", index=False)

print("Done! Saved as TWOSIDES_with_smiles.csv")

