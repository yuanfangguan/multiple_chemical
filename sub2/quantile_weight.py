import pandas as pd
import numpy as np
from scipy.stats import rankdata

def weighted_average_with_reference_quantile(file1_path, file2_path, weight1=0.5, weight2=0.5, output_file="weighted_quantile_normalized.csv"):
    # Read both files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Separate identifier
    stimulus_col = df1['stimulus']
    cols = df1.columns.drop('stimulus')

    # Weighted average
    weighted_values = df1[cols] * weight1 + df2[cols] * weight2

    # Prepare normalized result DataFrame
    normalized_values = pd.DataFrame(columns=cols)

    for col in cols:
        # Get reference values from file1
        reference_values = df1[col].sort_values().values

        # Get rank order in weighted_values[col]
        ranks = rankdata(weighted_values[col], method='ordinal') - 1  # zero-based index

        # Map weighted ranks to reference quantiles
        normalized_col = pd.Series(reference_values[ranks], index=weighted_values.index)

        normalized_values[col] = normalized_col

    # Combine stimulus column back
    normalized_values.insert(0, 'stimulus', stimulus_col)

    # Save to file and show as output
    normalized_values.to_csv(output_file, index=False)

    import ace_tools as tools; tools.display_dataframe_to_user(name="Weighted Quantile Normalized", dataframe=normalized_values)

    return normalized_values

# Example usage:
import sys
weighted_average_with_reference_quantile(sys.argv[1],sys.argv[2], weight1=0.7, weight2=0.3)

