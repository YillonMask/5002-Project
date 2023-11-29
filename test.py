from configuration import *
from beyes import *
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

ham_emails = []
for file_path in ham_files_path:
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
        ham_content_tokens = split_email(content)
        ham_emails.append(ham_content_tokens)

print('read ham folder successfully!')

spam_emails = []
for file_path in spam_files_path:
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
        spam_content_tokens = split_email(content)
        spam_emails.append(spam_content_tokens)

print('read spam folder successfully!')

word_prob_dict, letter_conditional_prob_dict= beye_train()

for tokens in ham_emails:
    classssification = beyes_test(word_prob_dict, letter_conditional_prob_dict, tokens)

    if classssification == 0:
        TRUE_NEGTIVE += 1
    else:
        FALSE_POSITIVE += 1

for tokens in spam_emails:
    classssification = beyes_test(word_prob_dict, letter_conditional_prob_dict, tokens)

    if classssification == 1:
        TRUE_POSITIVE += 1
    else:
        FALSE_NEGTIVE += 1

print(TRUE_POSITIVE, TRUE_NEGTIVE, FALSE_POSITIVE, FALSE_NEGTIVE)
print('the accuracy of the model is:', (TRUE_POSITIVE + TRUE_NEGTIVE) / (TRUE_POSITIVE + TRUE_NEGTIVE + FALSE_POSITIVE + FALSE_NEGTIVE))