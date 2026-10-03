import pandas as pd

# Load the CSV file
import sys
import sys
df = pd.read_csv(sys.argv[1])

# Clip all columns except the first ('stimulus') to the range [0, 5]
df.iloc[:, 1:] = df.iloc[:, 1:].clip(lower=0, upper=5)

# Save the clipped version back to a new CSV (or overwrite original)
df.to_csv(sys.argv[1]+"clipped.csv", index=False)

