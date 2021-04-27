import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Import data

df = pd.read_csv("C:\\Users\\internet\\Google Drive\\MTA\\Data Science applications for economists\\data\\train.csv")

# Feature Engineering

df = pd.get_dummies(df, columns = ["Sex"])

# Modeling

## Split data

my_vars = ["Parch","Sex_female","Sex_male"]

X = df[my_vars]

y = df["Survived"]


X_train,X_test, y_train , y_test = train_test_split(X, y)

## Train model

my_model = LogisticRegression()

my_model.fit(X = X_train, y = y_train)

## Predict 

y_pred = my_model.predict(X = X_test)


# Evaluate model

round(roc_auc_score(y_test,y_pred),4)

