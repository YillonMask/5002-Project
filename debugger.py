from configuration import * 
from helper import *
from beyes_class import *

train_database = 0
test_database = 0

uci_spambase = fetch_ucirepo(id=94)
train_database = pd.concat([uci_spambase.data.features, uci_spambase.data.targets], axis = 1, ignore_index=False)
print(train_database)

beyes_model = BasicNB()
#beyes_model.data_processing('./test database/spam', './test database/ham')
beyes_model.train(train_database)
beyes_model.test('./test database/spam', './test database/ham')