# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

#Loading training data
f = open("training_label.txt",'r',encoding = 'utf8')
lines = f.readlines()

x_train = []
y_train = []

for line in lines :
    y_train.append(int(line.split('+++$+++')[0]))
    x_train.append(line.split('+++$+++')[1].strip())

#Loading testing data
f = open("testing_label.txt",'r',encoding = 'utf8')
lines=f.readlines()

x_test = []
y_test = []

for line in lines :
    if len(line) == 1:
        continue
    y_test.append(int(line.split('#####')[0]))
    x_test.append(line.split('#####')[1].strip())	

#x_test = pd.DataFrame(x_test)
#y_test = pd.DataFrame(y_test)

#Word to vector, set stop words.
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(x_train[:5000]).toarray()
Y = np.array(y_train[:5000])

X_test = vectorizer.transform(x_test).toarray()
Y_test = np.array(y_test)

#adaboost
a = AdaBoostClassifier(n_estimators = 100, base_estimator = None,learning_rate = 1, random_state = 1)
a.fit(X,Y)
print(a.score(X,Y))

ada_predict = a.predict(X_test)
#print(ada_predict)
print(classification_report(Y_test, ada_predict, labels =[0,1]))

#xgbbost
xgbc = XGBClassifier().fit(X, Y)
print(xgbc.score(X, Y))

xgbc_predict = xgbc.predict(X_test)
print(classification_report(Y_test, xgbc_predict, labels =[0,1]))

