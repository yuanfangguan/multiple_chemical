import random
import pandas as pd

# Load the datasets
train1 = pd.read_csv('../../../sub2/data/raw/TASK2_Train_mixture_Dataset.csv')
train2 = pd.read_csv('../../data/raw/TASK2_Leaderboard_ActualValue.csv')

# Identify the common columns
common_columns = train1.columns.intersection(train2.columns)

# Subset both datasets to include only the common columns
train1_common = train1[common_columns]
train2_common = train2[common_columns]

# Merge the datasets on the common columns
merged_data = pd.merge(train1_common, train2_common, on=common_columns.tolist(), how='outer')

# Save the merged dataset
merged_data.to_csv('train_sub2.csv', index=False)

