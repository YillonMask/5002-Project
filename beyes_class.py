from configuration import *
from helper import *
from stopword import *

class BasicNB():
    def __init__(self) -> None:
        self.train_database = None
        self.word_prob = pd.DataFrame(columns=['spam', 'ham'])
        self.prior_prob_spam = 0
        self.prior_prob_ham = 0


    def train(self, train_database) -> None:
        self.train_database = train_database
        # get number of spam emails and number of ham emails
        number_spam_emails = 0
        number_ham_emails = 0

        # get the classification of each emails
        # 1 is spam emails and 0 is ham emails
        classification = self.train_database['Class'].values
        for value in classification:
            if int(value) == 1:
                number_spam_emails += 1
            elif int(value) == 0:
                number_ham_emails += 1
            else:
                raise ValueError('there is a error in classification of emails.')
        
        # get the prior prob of an email beging spam
        # P(spam) = number of spam emails / total number of emails
        self.prior_prob_spam = number_spam_emails / (number_spam_emails + number_ham_emails)
        # get the prior prob of an email beging ham
        # P(ham) = number of ham emails / total number of emails
        self.prior_prob_ham = number_ham_emails / (number_spam_emails + number_ham_emails)

        # split the dataframe into spam and ham
        word_prob_in_each_spam_email = self.train_database.groupby('Class').get_group(1)
        word_prob_in_each_ham_eamil = self.train_database.groupby('Class').get_group(0)
        # get the column titles of train database
        column_titles = self.train_database.columns
        # get the total prob of words in each type of email
        for title in column_titles[0: -4]:
            word_total_prob_in_spam = 0
            word_total_prob_in_ham = 0

            for prob_spam in word_prob_in_each_spam_email[title].values:
                word_total_prob_in_spam += prob_spam / number_spam_emails

            for prob_ham in word_prob_in_each_ham_eamil[title].values:
                word_total_prob_in_ham += prob_ham / number_ham_emails

            self.word_prob.loc[title.split('_')[2]] = [word_total_prob_in_spam, word_total_prob_in_ham]


    def data_processing(self, spam_folder_path, ham_folder_path):
        spam_emails, ham_emails = self.load_real_emails(spam_folder_path, ham_folder_path)
        # record totals words in train database without repitition
        number_spam_words = 0
        number_ham_words = 0
        total_words = []

        print(f'spam emails: {len(spam_emails)}, ham_emails: {len(ham_emails)}')
        counter = 0
        for spam_email in spam_emails:
            counter += 1
            print(counter)
            for spam_word in spam_email:
                if spam_word not in total_words and spam_word not in stop_words:
                    total_words.append(spam_word)
                number_spam_words += 1

        print('hello, I have finished load the spam word!')
        counter = 0
        for ham_email in ham_emails:
            counter += 1
            print(counter)
            for ham_word in ham_email:
                if ham_word not in total_words and ham_word not in stop_words:
                    total_words.append(ham_word)
                number_ham_words += 1

        # add a row call 'Class' to record the type of email
        # 1 is spam and 0 is ham
        total_words.append('Class')
        # set up a dataframe and string in list total_words are column names
        train_database = pd.DataFrame(columns = total_words)
        print(len(total_words))

        for spam_email in spam_emails:
            for spam_word in spam_email:
                new_row = pd.Series(0, index = train_database.columns)
                new_row[spam_word] = 1 / number_spam_words
                new_row['Class'] = 1
                train_database = train_database._append(new_row, ignore_index=True)

        for ham_email in ham_emails:
            for ham_word in ham_email:
                new_row = pd.Series(0, index = train_database.columns)
                new_row[ham_word] = 1 / number_ham_words
                new_row['Class'] = 0
                train_database = train_database._append(new_row, ignore_index=True)
        
        print(train_database)
        return train_database


    def load_real_emails(self, spam_folder_path, ham_folder_path):
        spam_emails = []
        ham_emails = []

        # get file names in the folder
        ham_files_path = get_files_path(ham_folder_path)
        spam_files_path = get_files_path(spam_folder_path)

        # get the words of each spam email
        for file_path in spam_files_path:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                spam_content_tokens = split_email(content)
                spam_emails.append(spam_content_tokens)

        print('read spam folder successfully!')

        # get the words of each ham email
        for file_path in ham_files_path:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                ham_content_tokens = split_email(content)
                ham_emails.append(ham_content_tokens)

        print('read ham folder successfully!')

        return spam_emails, ham_emails


    def classificatior(self, email):
        prob_spam = self.prior_prob_spam
        prob_ham = self.prior_prob_ham

        for word in email:
            if word in self.word_prob.index:
                prob_spam *= self.word_prob.loc[word, 'spam']
            if word in self.word_prob.index:
                prob_ham *= self.word_prob.loc[word, 'ham']

            
        if prob_ham > prob_spam:
            return 0
        else:
            return 1


    def test(self, spam_folder_path, ham_folder_path):
        print(self.word_prob)
        spam_emails, ham_emails = self.load_real_emails(spam_folder_path, ham_folder_path)
        # these are some parametre which can use to evaluate the model
        true_positive = 0
        true_negtive = 0
        false_positive = 0
        false_negtive = 0

        # identify different types of emails with our model
        for spam_email in spam_emails:
            classification = self.classificatior(spam_email)
            if classification == 1:
                true_positive += 1
            else:
                false_negtive += 1
            
        for ham_email in ham_emails:
            classification = self.classificatior(ham_email)
            if classification == 0:
                true_negtive += 1
            else:
                false_positive += 1

        print(
f'''
----------------------------------------------------------------
                            |               ACTUAL
                            -------------------------------------
                            |  Positive(spam)   Negtive(regular)
-----------------------------------------------------------------
        |   Positive(spam)  |        {true_positive}             {false_positive}
PREDICT | ------------------|------------------------------------
        |  Negtive(regular) |        {false_negtive}            {true_negtive}
-----------------------------------------------------------------
'''
        )
        print('the accuracy of the model is:', (true_positive + true_negtive) / (true_positive + true_negtive + false_positive + false_negtive))