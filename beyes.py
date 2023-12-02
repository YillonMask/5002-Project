from configuration import *
from helper import *

def beye_train():
    # fetch dataset 
    uci_spambase = fetch_ucirepo(id=94)

    '''
    calculate the total probability of these features, 
    except last three features related to capital letters,
    because I don't know how to use these three features in bayes model
    '''
    # get the classification(spam 1 or regular 0) of each row
    print(uci_spambase.data.targets)
    targets = uci_spambase.data.targets['Class'].values
    print('\nclassification: ', targets)
    
    # calculate the number of spam and regular letter
    spam_letter_counter = 0
    regular_letter_counter = 0

    for target in targets:
        if target == 1:
            spam_letter_counter += 1
        elif target == 0:
            regular_letter_counter += 1

    print('\nthe number of spam letter in UCI spambase is:', spam_letter_counter)
    print('\nthe number of regular letter in UCI spambase is:', regular_letter_counter)
    
    # get the title of each column (except last three columns)
    print(type(uci_spambase.data.features))
    column_titles = uci_spambase.data.features.columns[0:-3]
    print('\ncolumn_titles: ', column_titles)

    # calulate the P(word) and save it in a dict
    # probability = frequency / the total number of letter? maybe, I'm not sure hhhhhhh :)
    word_prob_dict = {}
    
    for title in column_titles:
        word_freq_array = uci_spambase.data.features[title].values
        for freq in word_freq_array:
            if title not in word_prob_dict:
                word_prob_dict[title.split('_')[-1]] = freq / (spam_letter_counter + regular_letter_counter)
            else:
                word_prob_dict[title.split('_')[-1]] += freq / (spam_letter_counter + regular_letter_counter)

    print('\nthe word_prob_dict is:', word_prob_dict)

    # calculate the fequence of each word or char in spam or regular letter and save them in a dict
    word_freq_dict = {}

    for title in column_titles:
        word_freq_array = uci_spambase.data.features[title].values
        for index, freq in enumerate(word_freq_array):
            # add label '_regular' and '_spam' to classify the frequency
            if targets[index] == 0:
                if title + '_regular' in word_freq_dict:
                    word_freq_dict[title.split('_')[-1] + '_regular'] += freq
                else:
                    word_freq_dict[title.split('_')[-1] + '_regular'] = freq
            elif targets[index] == 1:
                if title + '_spam' in word_freq_dict:
                    word_freq_dict[title.split('_')[-1] + '_spam'] += freq
                else:
                    word_freq_dict[title.split('_')[-1] + '_spam'] = freq

    print('\nthe word_freq_dict is:', word_freq_dict)

    # calculate the P(word|spam) or P(word|regular) and save them in a dict
    # probability = frequency / the total number of letter? maybe, I'm not sure hhhhhhh :)
    word_conditional_prob_dict = {}

    for key, value in word_freq_dict.items():
        if key.split('_')[-1] == 'spam':
            word_conditional_prob_dict[key] = value / spam_letter_counter
        if key.split('_')[-1] == 'regular':
            word_conditional_prob_dict[key] = value / regular_letter_counter

    print('\nthe word_conditional_prob_dict is:', word_conditional_prob_dict)

    # calculat the P(spam|word) and P(regular|word) accroding to Beyes theorem save them in a dict
    # P(spam|word) = P(word|spam) * P(spam) / P(word)
    # P(regular|word) = P(word|regular) * P(regular) / P(word)
    # P(word) save in word_prob_dict
    # P(word|spam) and P(word|regular) save in word_conditional_prob_dict
    prob_spam = spam_letter_counter / (spam_letter_counter + regular_letter_counter)
    letter_conditional_prob_dict = {}

    for key, value in word_conditional_prob_dict.items():
        letter_conditional_prob_dict[key] = value * prob_spam / word_prob_dict[key.split('_')[0]]

    print('\nthe letter_conditional_prob_dict is:', letter_conditional_prob_dict)

    return word_prob_dict, letter_conditional_prob_dict


def beyes_train_with_traindatabase(ham_train_folder, spam_train_folder):
    ham_train_filespath = get_files_path(ham_train_folder)
    spam_train_filespath = get_files_path(spam_train_folder)
    
    ham_emails = []
    for file_path in ham_train_filespath:
        with open(file_path, 'r', encoding='latin-1') as file:
            email= file.read()
            ham_email_tokens = set(split_email(email))
            ham_emails.append(ham_email_tokens)

    spam_emails = []
    for file_path in spam_train_filespath:
        with open(file_path, 'r', encoding='latin-1') as file:
            email = file.read()
            spam_email_tokens = set(split_email(email))
            spam_emails.append(spam_email_tokens)

    spam_letter_counter = 0
    regular_letter_counter = 0
    spam_tokens = 0
    regular_tokens = 0

    for email in spam_emails:
        spam_letter_counter += 1
        for token in email:
            spam_tokens += 1

    for email in ham_emails:
        regular_letter_counter += 1
        for token in email:
            regular_tokens += 1
    
    print(spam_tokens, regular_tokens)

    word_prob_dict = {}
    word_conditional_prob_dict = {}

    for email in spam_emails:
        for token in email:
            if token not in word_prob_dict:
                word_prob_dict[token] = 1 / (spam_tokens + regular_tokens)
            else:
                word_prob_dict[token] += 1 / (spam_tokens + regular_tokens)
            
            if token + ' spam' not in word_prob_dict:
                word_conditional_prob_dict[token + ' spam'] = 1 / (spam_tokens)
            else:
                word_conditional_prob_dict[token + ' spam'] += 1 / (spam_tokens)
    
    for email in ham_emails:
        for token in email:
            if token not in word_prob_dict:
                word_prob_dict[token] = 1 / (spam_tokens + regular_tokens)
            else:
                word_prob_dict[token] += 1 / (spam_tokens + regular_tokens)

            if token + ' regular' not in word_prob_dict:
                word_conditional_prob_dict[token + ' regular'] = 1 / (regular_tokens)
            else:
                word_conditional_prob_dict[token + ' regular'] += 1 / (regular_tokens)
    
    # calculat the P(spam|word) and P(regular|word) accroding to Beyes theorem save them in a dict
    # P(spam|word) = P(word|spam) * P(spam) / P(word)
    # P(regular|word) = P(word|regular) * P(regular) / P(word)
    # P(word) save in word_prob_dict
    # P(word|spam) and P(word|regular) save in word_conditional_prob_dict
    prob_spam = spam_letter_counter / (spam_letter_counter + regular_letter_counter)

    letter_conditional_prob_dict = {}

    for key, value in word_conditional_prob_dict.items():
        letter_conditional_prob_dict[key] = value * prob_spam / word_prob_dict[key.split(' ')[0]]

    print('\nthe word_prob_dict is:', len(word_prob_dict))
    print('\nthe letter_conditional_prob_dict is:', len(letter_conditional_prob_dict))

    return word_prob_dict, letter_conditional_prob_dict, spam_tokens, regular_tokens


def beyes_test(word_prob_dict, letter_conditional_prob_dict, email, spam_tokens, regular_tokens):
    # P(spam) = P(word_1)P(spam|word_1) + P(word_2)P(spam|word_2) + ··· + P(word_n)P(spam|word_n)
    # P(regular) = P(regular_1)P(regular|word_1) + P(word_2)P(regular|word_2) + ··· + P(word_n)P(regular|word_n)
    # if P(spam) > P(regular), then the letter is spam
    log_prob_spam = 0
    log_prob_regular = 0

    for token in email:
        key = token
        if key not in word_prob_dict:
            pass
        else:
            if key + ' spam' not in letter_conditional_prob_dict:
                log_prob_spam += word_prob_dict[key] * (1 / (spam_tokens))
            else:
                log_prob_spam += word_prob_dict[key] * letter_conditional_prob_dict[key + ' spam']
            if key + ' regular' not in letter_conditional_prob_dict:
                log_prob_regular += word_prob_dict[key] * (1 / (regular_tokens))
            else:
                log_prob_spam *= word_prob_dict[key] * letter_conditional_prob_dict[key + ' regular']
    
    print(log_prob_spam, log_prob_regular)
    if log_prob_spam > log_prob_regular:
        return 1
    else:
        return 0