"""
    CS5002 Fall 2023 SV
    Final Project
    Xinrui Yi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# fetch dataset
spambase = pd.read_csv(
    "/Users/xinruiyi/Documents/GitHub/5002-Project/spambase/spambase.data", header=None
)

# Change values greater than 0 to 1 in columns 0 to 53
spambase.iloc[:, 0:54] = np.where(spambase.iloc[:, 0:54] > 0, 1, spambase.iloc[:, 0:54])

# Split top 1813 rows into 20% test data and 80% training data
top_data = spambase.iloc[:1813, :]
top_train, top_test = train_test_split(top_data, test_size=0.2, random_state=42)

# Split the rest of the rows into 20% test data and 80% training data
rest_data = spambase.iloc[1813:, :]
rest_train, rest_test = train_test_split(rest_data, test_size=0.2, random_state=42)

# Calculate column average for the first 54 columns in top_train
column_avg_top = top_train.iloc[:, 0:54].mean()

# Print the column average
print("Column Average for the First 54 Columns in Top Train Data:")
print(column_avg_top)

# Calculate column average for the first 54 columns in top_train
column_avg_rest = rest_train.iloc[:, 0:54].mean()

# Print the column average
print("Column Average for the First 54 Columns in Rest Train Data:")
print(column_avg_rest)

# Combine column_avg_top and column_avg_rest into one table
column_avg_combined = pd.concat([column_avg_top, column_avg_rest], axis=1)
column_avg_combined.columns = ["Spam", "Ham"]

# Print the combined table
print("Combined Column Averages for the First 54 Columns:")
print(column_avg_combined)
