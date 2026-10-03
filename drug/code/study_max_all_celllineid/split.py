import pandas as pd
import sys
from sklearn.model_selection import train_test_split

# ================================
# Paths
# ================================
input_file = "../../data/summary_v_1_5_averaged.csv"
train_out = "train.csv"
test_out = "test.csv"

# ================================
# Load & Filter Data
# ================================
df = pd.read_csv(input_file)

# Filter out invalid drug names
filtered = df[
    (df["drug_row"].notna()) &
    (df["drug_col"].notna()) &
    (df["drug_row"].str.upper() != "NULL") &
    (df["drug_col"].str.upper() != "NULL")
].reset_index(drop=True)

print(f"✅ Loaded {len(df)} rows total; {len(filtered)} valid after filtering")

# ================================
# Split by study_name
# ================================
if "study_name" not in filtered.columns:
    raise ValueError("❌ Column 'study_name' not found in input file!")

unique_studies = filtered["study_name"].dropna().unique()
print(f"📚 Found {len(unique_studies)} unique study names")

# Split study names 80/20
train_studies, test_studies = train_test_split(
    unique_studies,
    test_size=0.2,
    random_state=int(sys.argv[1]) if len(sys.argv) > 1 else 42
)

train_df = filtered[filtered["study_name"].isin(train_studies)]
test_df = filtered[filtered["study_name"].isin(test_studies)]

# ================================
# Save Results
# ================================
train_df.to_csv(train_out, index=False)
test_df.to_csv(test_out, index=False)

print(f"✅ Saved {len(train_df)} rows to {train_out} ({len(train_studies)} studies)")
print(f"✅ Saved {len(test_df)} rows to {test_out} ({len(test_studies)} studies)")

