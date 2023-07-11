#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import joblib

os.getcwd()

os.chdir("E:/Data Science/Data")

titanic_train = pd.read_csv("train.csv")

titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)


X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier(criterion = 'entropy')

dt_grid = {'criterion':['gini','entropy'], 'max_depth':list(range(3,12)), 'min_samples_split':[2,3,6,7,8]}
param_grid = model_selection.GridSearchCV(dt, dt_grid, cv=10) 
param_grid.fit(X_train, y_train) 
print(param_grid.best_score_) 
print(param_grid.best_params_)
print(param_grid.score(X_train, y_train)) 

os.getcwd()
joblib.dump(param_grid, "TitanicModel.pkl")


