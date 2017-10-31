#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:47:10 2017

@author: sziliotti
"""

# Kaggle: Titanic competition

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
train = pd.read_csv('../raw_data/train.csv')
test = pd.read_csv('../raw_data/test.csv')

# Dropping unecessary features
train = train.drop(['PassengerId','Ticket','Cabin','Name','Embarked'], axis=1)
PassengerId = test['PassengerId']
test = test.drop(['Name','Ticket','Cabin','Name','Embarked','PassengerId'], axis=1)

# Filling missing values in the training set, with the most occurency
test['Fare'].fillna(test['Fare'].median(), inplace = True)
train['Age'].fillna(train['Age'].mean(), inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)

# Getting childs 
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

train['Person'] = train[['Age','Sex']].apply(get_person, axis=1)
test['Person'] = test[['Age','Sex']].apply(get_person, axis=1)

train.drop(['Sex'],axis = 1, inplace = True)
test.drop(['Sex'],axis = 1, inplace = True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(train['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_titanic)
test = test.join(person_dummies_test)

train.drop(['Person'], axis = 1, inplace = True)
test.drop(['Person'], axis = 1, inplace = True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=5)
pclass_dummies = pd.get_dummies(train['Pclass'])
pclass_dummies.columns = ['class1','class2','class3']
pclass_dummies.drop(['class3'], axis=1, inplace = True)

pclass_dummies_train = pd.get_dummies(train['Pclass'])
pclass_dummies_train.columns = ['class1','class2','class3']
pclass_dummies_train.drop(['class3'], axis=1, inplace = True)

pclass_dummies_test = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['class1','class2','class3']
pclass_dummies_test.drop(['class3'], axis=1, inplace = True)

train.drop(['Pclass'], axis=1, inplace = True)
test.drop(['Pclass'], axis=1, inplace = True)

# Handling family members
train['Family'] = train['SibSp'] + train['Parch']
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] = test['SibSp'] + test['Parch']
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train.drop(['SibSp','Parch'],axis=1, inplace=True)
test.drop(['SibSp','Parch'],axis=1, inplace=True)

# Getting the training data and target variable
y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)
X_test = test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing libraries for classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Creating Classifiers
clf_svm = SVC(kernel='linear', random_state=42).fit(X_train, y_train)
clf_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)
clf_LR = LogisticRegression(random_state = 42).fit(X_train, y_train)
clf_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski').fit(X_train, y_train)

clf_svm.score(X_train, y_train)
clf_RF.score(X_train, y_train)
clf_LR.score(X_train, y_train)
clf_knn.score(X_train, y_train)

# Predicting for test set
y_pred_svm = clf_svm.predict(X_test)
y_pred_RF = clf_RF.predict(X_test)
y_pred_LR = clf_LR.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)

#Saving into csv file for submission
results_svm = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred_svm}).to_csv("../results/results_svm.csv", index=None)
results_RF = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred_RF}).to_csv("../results/results_RF.csv", index=None)
results_LR = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred_LR}).to_csv("../results/results_LR.csv", index=None)
results_knn = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred_knn}).to_csv("../results/results_knn.csv", index=None)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = clf_svm,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
y_pred_gs_svm = grid_search.predict(X_test)

results_gs_svm = pd.DataFrame({"PassengerId": PassengerId, "Survived": y_pred_gs_svm}).to_csv("../results/results_gs_svm.csv", index=None)


