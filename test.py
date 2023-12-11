from beyes import *
from helper import *


ham_folder_path = './test database/ham'
spam_folder_path = './test database/spam'
ham_files_path = get_files_path(ham_folder_path)
spam_files_path = get_files_path(spam_folder_path)

X = []
Y = []
for file_path in ham_files_path:
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
        X.append(content)
        Y.append(0)

for file_path in spam_files_path:
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
        X.append(content)
        Y.append(1)

clf, vectorizer = spam_filter_train(X, Y)
