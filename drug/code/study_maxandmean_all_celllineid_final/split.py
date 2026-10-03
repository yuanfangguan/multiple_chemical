import pandas as pd

# ================================
# Paths
# ================================
input_file = "../../data/summary_v_1_5_averaged.csv"
train_out = "train.csv"

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

# Optional: check study_name exists (not needed for splitting anymore)
if "study_name" not in filtered.columns:
    print("⚠️ Warning: Column 'study_name' not found, continuing anyway.")

# ================================
# Save Single Training File
# ================================
filtered.to_csv(train_out, index=False)
print(f"📁 Saved {len(filtered)} rows to {train_out}")

