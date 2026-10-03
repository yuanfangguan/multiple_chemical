#!/usr/bin/env python3
import pandas as pd
import numpy as np

# 1. Load the raw data
# Fixed column name 'rxnorn' and added low_memory=False
file_path = "TWOSIDES.csv"
df = pd.read_csv(file_path, low_memory=False)

print(f"Initial row count: {len(df):,}")

# 2. Fix the column name in the dataframe so it's consistent
# This maps the typo 'rxnorn' to the correct 'rxnorm'
df = df.rename(columns={'drug_1_rxnorn_id': 'drug_1_rxnorm_id'})

# 3. Convert aggregation columns to numeric
# Using 'float32' saves significant RAM compared to the default float64
cols_to_fix = ['PRR', 'mean_reporting_frequency', 'A', 'B', 'C', 'D']

for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

# 4. Drop invalid rows
# We drop rows where the math columns are NaN or the names are missing
df = df.dropna(subset=cols_to_fix + ["drug_1_concept_name", "drug_2_concept_name", "condition_concept_name"])

print(f"Rows remaining after cleaning: {len(df):,}")

# 5. Grouping
# We use the corrected name 'drug_1_rxnorm_id' here
group_cols = [
    "drug_1_rxnorm_id", 
    "drug_1_concept_name", 
    "drug_2_rxnorm_id", 
    "drug_2_concept_name", 
    "condition_concept_name"
]

print("Starting aggregation (this may take a minute for 42M rows)...")
df_grouped = df.groupby(group_cols, as_index=False).agg({
    'PRR': 'mean',
    'mean_reporting_frequency': 'mean',
    'A': 'sum',
    'B': 'sum',
    'C': 'sum',
    'D': 'sum'
})

# Add a unique row ID for the final output
df_grouped["row_id"] = range(len(df_grouped))

print(f"Final grouped row count: {len(df_grouped):,}")

# 6. Save the cleaned, grouped data
output_file = "TWOSIDES_drugpair_clean_MRF_grouped.csv"
df_grouped.to_csv(output_file, index=False)
print(f"✔ Success! Saved to {output_file}")
