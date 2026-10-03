import pandas as pd

df = pd.read_csv("../../data/TWOSIDES_with_smiles.csv", usecols=[
    "drug_1_rxnorn_id", "drug_2_rxnorm_id",
    "condition_meddra_id"
])

print("Total rows:", len(df))
print("Unique triplets:", df.drop_duplicates().shape[0])
print("Duplicate rows:", len(df) - df.drop_duplicates().shape[0])

