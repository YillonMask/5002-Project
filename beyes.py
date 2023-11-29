from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
uci_spambase = fetch_ucirepo(id=94)

'''
calculate the total probability of these features, 
except last three features related to capital letters,
because I don't know how to use these three features in bayes model
'''
# get the title of each column (except last three columns)
print(uci_spambase.data.features)
column_titles = uci_spambase.data.features.columns[0:-3]
print('\ncolumn_titles: ', column_titles)

# get the classification(spam 1 or regular 0) of each row
print(uci_spambase.data.targets)
targets = uci_spambase.data.targets['Class'].values
print('\nclassification: ', targets)

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

# calulate the P(word) and save it in a dict
# probability = frequency / the total number of letter? maybe, I'm not sure hhhhhhh :)
word_prob_dict = {}
for key, value in word_freq_dict.items():
    word_prob_dict[key] = value / (spam_letter_counter + regular_letter_counter)

print('\nthe dict record the probability of each word is:', word_prob_dict)

# calculate the P(word|spam) or P(word|regular) and save them in a dict
# probability = frequency / the total number of letter? maybe, I'm not sure hhhhhhh :)
word_conditional_prob_dict = {}

for key, value in word_freq_dict.items():
    if key.split('_')[3] == 'spam':
        word_conditional_prob_dict[key] = value / spam_letter_counter
    if key.split('_')[3] == 'regular':
        word_conditional_prob_dict[key] = value / regular_letter_counter

print('\nthe dict record the conditional probability of each word in spam or regular letter is:', word_conditional_prob_dict)
