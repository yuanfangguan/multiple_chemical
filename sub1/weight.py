import pandas as pd

import sys
# Set file paths
file1 = sys.argv[1]
file2 = sys.argv[2]

# Load both files
df1 = pd.read_csv(file1, index_col=0)
df2 = pd.read_csv(file2, index_col=0)

# Ensure both files have the same structure
print(df1.shape, df2.shape)
assert df1.shape == df2.shape
assert all(df1.columns == df2.columns)
assert all(df1.index == df2.index)

# Set the weight for the first file
w = 0.8  # for example, 60% file1, 40% file2

# Weighted average
df_weighted = w * df1 + (1 - w) * df2

# Restore the index name and save
df_weighted.index.name = 'stimulus'
df_weighted.to_csv('predictions.csv')

