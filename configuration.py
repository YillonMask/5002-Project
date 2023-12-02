import nltk
from nltk.tokenize import word_tokenize
from ucimlrepo import fetch_ucirepo
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd

# execute only once
# nltk.download('punkt')