"""
    CS5002 Fall 2023 SV
    Final Project
    Xinrui Yi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# fetch dataset
spambase = pd.read_csv("./spambase/spambase.data", header=None)

# Change values greater than 0 to 1 in columns 0 to 53
spambase.iloc[:, 0:54] = np.where(spambase.iloc[:, 0:54] > 0, 1, spambase.iloc[:, 0:54])

# Split top 1813 rows into 20% test data and 80% training data
top_data = spambase.iloc[:1813, :]
top_train, top_test = train_test_split(
    top_data, test_size=0.1, train_size=0.9, random_state=42
)

# Split the rest of the rows into 20% test data and 80% training data
rest_data = spambase.iloc[1813:, :]
rest_train, rest_test = train_test_split(
    rest_data, test_size=0.1, train_size=0.9, random_state=42
)

# Calculate column average for the first 54 columns in top_train
column_avg_top = top_train.iloc[:, 0:54].mean()

# Print the column average
# print("Column Average for the First 54 Columns in Top Train Data (Spam):")
# print(column_avg_top)

# Calculate column average for the first 54 columns in top_train
column_avg_rest = rest_train.iloc[:, 0:54].mean()

# Print the column average
# print("Column Average for the First 54 Columns in Rest Train Data (Ham):")
# print(column_avg_rest)

# Combine column_avg_top and column_avg_rest into one table
column_avg_combined = pd.concat([column_avg_top, column_avg_rest], axis=1)
column_avg_combined.columns = ["Spam", "Ham"]

# Print the combined table
# print("Combined Column Averages for the First 54 Columns:")
# print(column_avg_combined)

# calculate the probability of spam emails in the training data
spam_probability = len(top_train[top_train[57] == 1]) / (
    len(top_train) + len(rest_train)
)
print("Probability of Spam Emails in Training Data:", spam_probability)

# calculate the probability of ham emails in the training data
ham_probability = len(rest_train[rest_train[57] == 0]) / (
    len(top_train) + len(rest_train)
)
print("Probability of Ham Emails in Training Data:", ham_probability)


# test the model
# for email in test data, calculate the probability of being spam
# Multiply the probability of spam for all words with value 1 in the test data
spam_product = spam_probability
ham_product = ham_probability
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# test top_test data

for index, row in top_test.iterrows():
    for word in top_test.columns[:54]:
        if row[word] == 1:
            ham_product *= column_avg_rest[word]
            spam_product *= column_avg_top[word]
    if spam_product > ham_product:
        true_positive += 1
    else:
        false_negative += 1

# test rest_test data
for index, row in rest_test.iterrows():
    for word in rest_test.columns[:54]:
        if row[word] == 1:
            ham_product *= column_avg_rest[word]
            spam_product *= column_avg_top[word]
    if spam_product > ham_product:
        false_positive += 1
    else:
        true_negative += 1

# calculate the accuracy
accuracy = (true_positive + true_negative) / (
    true_positive + true_negative + false_positive + false_negative
)

precision = true_positive / (true_positive + false_positive)

print(f"true positive: {true_positive}")
print(f"true negative: {true_negative}")
print(f"false positive: {false_positive}")
print(f"false negative: {false_negative}")
print("Accuracy:", accuracy)
print("Precision:", precision)
