from configuration import *
from helper import *

global TRUE_POSITIVE
global TRUE_NEGTIVE
global FALSE_POSITIVE
global FALSE_NEGTIVE

TRUE_POSITIVE = 0
TRUE_NEGTIVE = 0
FALSE_POSITIVE = 0
FALSE_NEGTIVE = 0

ham_folder_path = './test database/ham'
spam_folder_path = './test database/spam'
ham_files_path = get_files_path(ham_folder_path)
spam_files_path = get_files_path(spam_folder_path)

for file_path in ham_files_path:
    with open(file_path, 'r') as file:
        content = file.read()
        content_token = split_email(content)

for file_path in spam_files_path:
    with open(file_path, 'r') as file:
        content = file.read()