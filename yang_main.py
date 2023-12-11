# File -> Text -> Data
import os  # For file operations
import pandas as pd  # For data processing
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.feature_extraction.text import CountVectorizer  # For converting text to numerical data
# Model
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes model for classification
from sklearn.metrics import classification_report  # For performance analysis

# 1 Read files (handle different encodings) -> emails
def read_emails(folder):
    emails = []
    for filename in os.listdir(folder):
        try:
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                emails.append(file.read())
        except UnicodeDecodeError:
            try:
                with open(os.path.join(folder, filename), 'r', encoding='ISO-8859-1') as file:
                    emails.append(file.read())
            except:
                pass  # Ignore files that can't be read
    return emails

# Reading data
ham_folder = '/Users/yangyang/Desktop/5002-Project-main/test database/ham'
spam_folder = '/Users/yangyang/Desktop/5002-Project-main/test database/spam'

ham_emails = read_emails(ham_folder)
spam_emails = read_emails(spam_folder)

# 2 Create DataFrame
df = pd.DataFrame({
    'email': ham_emails + spam_emails,
    'label': ['ham'] * len(ham_emails) + ['spam'] * len(spam_emails)
})

# 3 Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2, random_state=42)

# 4 Feature extraction
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_counts, y_train)
# Predict and evaluate
y_pred = model.predict(X_test_counts)
print(classification_report(y_test, y_pred))
