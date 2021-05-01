import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import re
import os

# Import data

df_raw = pd.read_csv("C:" + os.environ["HOMEPATH"] + "\\Google Drive\\MTA\\Data Science applications for economists\\data\\train.csv")

#Impute NA

# impute age by averages by pclass

df_raw["Age"] = df_raw.groupby("Pclass")["Age"].transform(lambda x:x.fillna(x.mean()))

# impute cabin by most frequent in "cheap" decks

df_raw["Cabin_imp"] = df_raw["Cabin"].astype(str).apply(lambda x:x[0])

df_raw["Cabin_imp"].unique()

df_raw[["Cabin_imp","Pclass"]].value_counts()

df_raw["Cabin_imp"] = df_raw["Cabin_imp"].replace("n","F")    

# impute embarked by most frequent

imp = SimpleImputer(strategy="most_frequent")

df_raw["Embarked"] = imp.fit_transform(np.asarray(df_raw["Embarked"]).reshape(-1,1))
