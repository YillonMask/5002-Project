"""
    CS5002 Fall 2023 SV
    Final Project
    Xinrui Yi
"""
from ucimlrepo import fetch_ucirepo
import pandas

# fetch dataset
spambase = pandas.read_csv('/Users/xinruiyi/Documents/GitHub/5002-Project/spambase/spambase.data')

# data (as pandas dataframes)
X = spambase.data.features
y = spambase.data.targets

# metadata
print(spambase.metadata)

# variable information
print(spambase.variables)