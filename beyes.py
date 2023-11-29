from configuration import *


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
    print(uci_spambase.data.features)
    column_titles = uci_spambase.data.features.columns[0:-3]
    print('\ncolumn_titles: ', column_titles)

    # calulate the P(word) and save it in a dict
    # probability = frequency / the total number of letter? maybe, I'm not sure hhhhhhh :)
    word_prob_dict = {}
    
    for title in column_titles:
        word_freq_array = uci_spambase.data.features[title].values
        for freq in word_freq_array:
            if title not in word_prob_dict:
                word_prob_dict[title] = freq / (spam_letter_counter + regular_letter_counter)
            else:
                word_prob_dict[title] += freq / (spam_letter_counter + regular_letter_counter)

    print('\nthe word_prob_dict is:', word_prob_dict)

    # calculate the fequence of each word or char in spam or regular letter and save them in a dict
    word_freq_dict = {}

    for title in column_titles:
        word_freq_array = uci_spambase.data.features[title].values
        for index, freq in enumerate(word_freq_array):
            # add label '_regular' and '_spam' to classify the frequency
            if targets[index] == 0:
                if title + '_regular' in word_freq_dict:
                    word_freq_dict[title + '_regular'] += freq
                else:
                    word_freq_dict[title + '_regular'] = freq
            elif targets[index] == 1:
                if title + '_spam' in word_freq_dict:
                    word_freq_dict[title + '_spam'] += freq
                else:
                    word_freq_dict[title + '_spam'] = freq

    print('\nthe dict record the fequency of each word in spam or regular letter:', word_freq_dict)

    # calculate the P(word|spam) or P(word|regular) and save them in a dict
    # probability = frequency / the total number of letter? maybe, I'm not sure hhhhhhh :)
    word_conditional_prob_dict = {}

    for key, value in word_freq_dict.items():
        if key.split('_')[3] == 'spam':
            word_conditional_prob_dict[key] = value / spam_letter_counter
        if key.split('_')[3] == 'regular':
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
        letter_conditional_prob_dict[key] = value * prob_spam / word_prob_dict[key.split('_')[0] + '_' + key.split('_')[1] + '_' + key.split('_')[2]]

    print('\nthe letter_conditional_prob_dict is:', letter_conditional_prob_dict)

    return word_prob_dict, letter_conditional_prob_dict


def beyes_test(word_prob_dict, letter_conditional_prob_dict, content_tokens):
    # P(spam) = P(word_1)P(spam|word_1) + P(word_2)P(spam|word_2) + ··· + P(word_n)P(spam|word_n)
    # P(regular) = P(regular_1)P(regular|word_1) + P(word_2)P(regular|word_2) + ··· + P(word_n)P(regular|word_n)
    # if P(spam) > P(regular), then the letter is spam
    prob_spam = 0
    prob_regular = 0

    for token in content_tokens:
        key = 'word_freq_' + token

        if key not in word_prob_dict:
            pass
        else:
            prob_spam += word_prob_dict[key] * letter_conditional_prob_dict[key + '_spam']
            prob_regular += word_prob_dict[key] * letter_conditional_prob_dict[key + '_regular']

    if prob_spam > prob_regular:
        return 1
    else:
        return 0