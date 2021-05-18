import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
#from sklearn.preprocessing import OneHotEncoder as cat_coder
from sklearn.metrics import roc_auc_score

import os

# Import data

df_raw = pd.read_csv("C:" + os.environ["HOMEPATH"] + "\\Google Drive\\MTA\\Data Science applications for economists\\data\\train.csv")

df = df_raw.drop(["Ticket","Cabin","Age","Name"],axis = 1)

df = df.dropna()

# Feature Engineering

## Categorical variables encoding

df = pd.get_dummies(df, columns=["Sex","Embarked","Pclass"])



# Modeling

## Split data

X = df.drop(["Survived"], axis = 1)

y = df["Survived"]


X_train,X_test, y_train , y_test = train_test_split(X, y)

## Train model

tree_model = DecisionTreeClassifier()

tree_model.fit(X = X_train, y = y_train)

## Predict 

y_pred = tree_model.predict(X = X_test)


# Evaluate model

round(roc_auc_score(y_test,y_pred),4)

# Visualize top tree

plot_tree(tree_model,max_depth=1)

