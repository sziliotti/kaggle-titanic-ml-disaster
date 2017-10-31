#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:18:36 2017

@author: sziliotti
"""

#Data preprocessing

#Importando as libs necessárias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importando os datasets
datasetTRAIN = pd.read_csv("../input/train.csv")
datasetTEST = pd.read_csv("../input/test.csv")

X_train = datasetTRAIN.iloc[:,2:-4] # Deve ser um Dataframe
X_train = X_train.drop(['Name'], axis=1)
X_test = datasetTEST.iloc[:,1:-4] # Deve ser um Dataframe
X_test = X_test.drop(['Name'], axis=1)

#Tratando categorical data by Pandas
X_train['Sex'] = X_train['Sex'].map({'male': 1, 'female': 0})
X_test['Sex'] = X_test['Sex'].map({'male': 1, 'female': 0})


#Dividindo em conjunto de treino e conjunto de teste
X_train = X_train.values
y_train = datasetTRAIN.iloc[:,1].values

X_test = X_test.values


#Tratando missing data (Campo Age)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:,2:3])
X_train[:,2:3] = imputer.transform(X_train[:,2:3])

imputer = imputer.fit(X_test[:,3:4])
X_test[:,2:3] = imputer.transform(X_test[:,2:3])



##Feature Scaling (Standariza)
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)




# CRIANDO OS MODELOS 
# Testando o modelo - RANDOM FOREST CLASSIFICATION 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) # n_estimators are the numbers of trees
regressor.fit(X_train, y_train)

# Prevendo o novo resultado
y_pred = regressor.predict(6.5)



# Testando modelo - KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Prevendo os resultados no dataset de Test.
y_pred = classifier.predict(X_test)


# Criando a matriz de confusão
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# Visualização do Random Forest Regression 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






##### OUTRAS SOLUCOES
