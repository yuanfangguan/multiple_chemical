import random
import sys
import pandas as pd

# Set the random seed
random.seed(int(sys.argv[1]))

# Open the output training file
TRAIN = open('train.csv', 'w')

# Load the datasets
train1 = pd.read_csv('../../data/raw/TASK1_training.csv')
train2 = pd.read_csv('../../data/raw/TASK1_Leaderboard_ActualValue.csv')

# Identify the common columns
common_columns = train1.columns.intersection(train2.columns)

# Subset both datasets to include only the common columns
train1_common = train1[common_columns]
train2_common = train2[common_columns]

# Merge the datasets on the common columns
merged_data = pd.merge(train1_common, train2_common, on=common_columns.tolist(), how='outer')

# Save the merged dataset
merged_data.to_csv('train.csv', index=False)

# Close the training file
TRAIN.close()

# Open the output test file
TEST = open('test.csv', 'w')

# Read and write lines from the test submission file
FILE = open('../../data/raw/TASK1_test_set_Submission_form.csv', 'r')
for line in FILE:
    TEST.write(line)

# Close the test file
TEST.close()

