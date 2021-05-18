import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import re
import os

# Import data

df_raw = pd.read_csv("C:" + os.environ["HOMEPATH"] + "\\Google Drive\\MTA\\Data Science applications for economists\\data\\train.csv")

df = df_raw.copy()

#Impute NA

# impute age by averages by pclass

df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x:x.fillna(x.mean()))

# impute cabin by most frequent in "cheap" decks

df["Cabin"] = df["Cabin"].astype(str).apply(lambda x:x[0])

df["Cabin"].unique()

df[["Cabin","Pclass"]].value_counts()

df["Cabin"] = df["Cabin"].replace("n","F")    

# impute embarked by most frequent

imp = SimpleImputer(strategy="most_frequent")

df["Embarked"] = imp.fit_transform(np.asarray(df["Embarked"]).reshape(-1,1))

# Encode nominal values

df = pd.get_dummies(df, columns=["Sex","Embarked","Cabin","Pclass"])
