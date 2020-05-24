#http://scikit-learn.org/stable/modules/tree.html

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from IPython.display import display
import pydotplus
import re
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

#(1) Read the data
train = pd.read_csv('C:/Users/Zach/Desktop/Data Mining/HW1/sf-crime/train.csv')
#(2-1) Replace the missing value with zero
train.fillna(0)
#Delete unnecessary columns
train = train.drop(['Address','Resolution','Descript'], axis = 1)

#(2-3) Extract the hour of time from column 'Dates' by regular expression 
train['Time'] = train['Dates']
train['Time'] = train['Time'].replace(to_replace = '[0-9]{4}\-[0-9]{2}\-[0-9]{2}\s', value = '', regex = True)
train['Time'] = train['Time'].replace(to_replace = '\:[0-9]{2}\:[0-9]{2}', value = '', regex = True)

#Delete the "Date" column
train = train.drop('Dates',axis = 1)
#Transform the variables into dummies
train = pd.get_dummies(train, columns = ['Time','PdDistrict','DayOfWeek'])

#Random split train set and test set
train_target = train['Category']
train_data = train.drop(['Category'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(train_data,train_target,
	test_size = 0.25, random_state = 666)

#Build the classifier
dt = tree.DecisionTreeClassifier(random_state=0, max_depth = 4)
dt_tree = dt.fit(X_train, y_train)
tree.plot_tree(dt_tree)

#Prediction
test_y_predicted = dt_tree.predict(X_test)
print(test_y_predicted)

#Answer
print(y_test)

#Accuracy, presicion, and recall
print(classification_report(y_test, test_y_predicted, target_names = list(train['Category'].value_counts().index)))

#train.to_csv('C:/Users/Zach/Desktop/Data Mining/HW1/sf-crime/LetMeSee.csv')
#print(train)
