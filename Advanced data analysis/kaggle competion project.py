import pandas as pd
import numpy as np
import xgboost as xgb
import re

train_dataset = pd.read_csv('train_dataset.csv')
train_label = pd.read_csv('train_label.csv')
test_dataset = pd.read_csv('test_dataset.csv')
test_id = test_dataset['encounter_id']

train_dataset = train_dataset.drop(['encounter_id','patient_nbr','weight','payer_code','medical_specialty','metformin.pioglitazone','citoglipton','examide'], axis = 1)
train_dataset['age'] = train_dataset['age'].replace(to_replace = '\[', value = '', regex = True)
test_dataset = test_dataset.drop(['encounter_id','patient_nbr','weight','payer_code','medical_specialty','metformin.pioglitazone','citoglipton','examide'], axis = 1)
test_dataset['age'] = test_dataset['age'].replace(to_replace = '\[', value = '', regex = True)

train_objs_num = len(train_dataset)
dataset = pd.concat(objs=[train_dataset, test_dataset], axis=0)
dataset = pd.get_dummies(dataset, columns = ["race",'gender','age','admission_type_id',\
                                                             'discharge_disposition_id','admission_source_id',\
                                                             'diag_1','diag_2',\
                                                             'diag_3','max_glu_serum','A1Cresult',\
                                                             'change','diabetesMed','metformin','repaglinide','nateglinide',\
                                                             'chlorpropamide','glimepiride','acetohexamide','glipizide',\
                                                             'glyburide','tolbutamide','pioglitazone','rosiglitazone'\
                                                             ,'acarbose','miglitol','troglitazone','tolazamide',\
                                                             'insulin','glyburide.metformin','glipizide.metformin'\
                                                             ,'glimepiride.pioglitazone','metformin.rosiglitazone'])
train_dataset = dataset[:train_objs_num]
test_dataset = dataset[train_objs_num:]


train_label['readmitted'] = train_label['readmitted'].replace('<30',0)
train_label['readmitted'] = train_label['readmitted'].replace('NO',1)
train_label['readmitted'] = train_label['readmitted'].replace('>30',2)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = train_dataset  #independent columns
y = train_label.iloc[:,1]   #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=2413)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(100,'Score'))  #print 10 best features

param = list(featureScores.nlargest(1800,'Score')['Specs'])

train_label = pd.read_csv('train_label.csv')
train_label = train_label.drop('encounter_id', axis = 1)

df = train_dataset[param]
dff = test_dataset[param]

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
XX_train, XX_test, yy_train, yy_test = train_test_split(df[:20000],train_label[:20000],
	test_size = 0.2, random_state = 0)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
s = SVC(kernel = 'linear',decision_function_shape='ova').fit(train_dataset, train_label.values.ravel())
