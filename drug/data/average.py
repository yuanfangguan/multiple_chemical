import pandas as pd
import numpy as np

input_file = "summary_v_1_5.csv"
output_file = "summary_v_1_5_averaged.csv"

# ========================
# Load safely (avoid DtypeWarning)
# ========================
df = pd.read_csv(input_file, low_memory=False)

# ========================
# Clean up missing / invalid drug names
# ========================
def clean_val(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().upper()
    if x in ["NULL", "NAN", "NA", "\\N", "NONE", ""]:
        return np.nan
    return x

df["drug_row"] = df["drug_row"].map(clean_val)
df["drug_col"] = df["drug_col"].map(clean_val)

# Drop rows missing either drug
df = df.dropna(subset=["drug_row", "drug_col"])
print(f"✅ Remaining after cleaning: {len(df)} rows")

# ========================
# Canonical pair key (unordered)
# ========================
def canonical_pair(a, b):
    return tuple(sorted([str(a), str(b)]))

df["pair"] = df.apply(lambda x: canonical_pair(x["drug_row"], x["drug_col"]), axis=1)
df[["drugA", "drugB"]] = pd.DataFrame(df["pair"].tolist(), index=df.index)

# ========================
# Identify numeric columns for averaging
# ========================
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
agg_funcs = {col: "mean" for col in numeric_cols}

# ========================
# Group by BOTH drugs and cell line
# ========================
group_keys = ["drugA", "drugB", "cell_line_name"]

grouped_numeric = df.groupby(group_keys, as_index=False).agg(agg_funcs)

# Keep representative metadata (non-numeric)
meta_cols = [c for c in df.columns if c not in numeric_cols + ["pair", "drugA", "drugB"]]
meta_df = (
    df.groupby(group_keys, as_index=False)
      .agg({col: "first" for col in meta_cols})
)

# Merge metadata + averaged numeric columns
final_df = pd.merge(meta_df, grouped_numeric, on=group_keys, how="inner")

# ========================
# Save
# ========================
final_df.to_csv(output_file, index=False)
print(f"✅ Averaged file saved to: {output_file}")
print(f"🧪 Resulting shape: {final_df.shape}")

