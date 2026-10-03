import pandas as pd

# Load the CSV file
df = pd.read_csv('predictions.csv')

# Find the maximum and minimum values for each column
max_values = df.max()
min_values = df.min()

# Print out the results
print("Maximum Values for each feature:")
print(max_values)

print("\nMinimum Values for each feature:")
print(min_values)

