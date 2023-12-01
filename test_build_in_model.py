from configuration import *
from helper import *

'''
uci_spambase = fetch_ucirepo(id=94)
X = uci_spambase.data.features
Y = uci_spambase.data.targets

X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.75,test_size=0.25,random_state=42,shuffle=True)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

multinomial = MultinomialNB(alpha=1.2,force_alpha=True,fit_prior=False)
multinomial.fit(X_train,y_train)
y_pred = multinomial.predict(X_test)

accuracy = np.mean(y_pred == y_test.values.ravel())
print(accuracy)
'''

