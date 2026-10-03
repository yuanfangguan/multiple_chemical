import pandas as pd

# Function to check for NaN values in a DataFrame and print their positions
def check_nan_in_dataframe(df, name="DataFrame"):
    # Find rows and columns with NaN values
    nan_positions = df[df.isna().any(axis=1)]
    
    if not nan_positions.empty:
        print(f"NaN values found in {name} at the following positions:")
        for row in nan_positions.index:
            nan_columns = nan_positions.loc[row].isna()
            for column in nan_columns.index[nan_columns]:
                print(f"Row {row}, Column: '{column}'")
    else:
        print(f"No NaN values found in {name}.")

# Example usage with your dataset
df_train = pd.read_csv("train.csv")  # Load your dataset (replace with your path)
check_nan_in_dataframe(df_train, "Training Data")

