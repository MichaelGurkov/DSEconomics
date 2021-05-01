import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import *
import os

# Import data

df_raw = pd.read_csv("C:" + os.environ["HOMEPATH"] + "\\Google Drive\\MTA\\Data Science applications for economists\\data\\train.csv")

df_raw.head()

df_raw.describe()

df_raw.info()

# Check class balance

sns.countplot(x = "Survived", data = df_raw)

sns.countplot(x = "Survived", data = df_raw, hue = "Pclass").set_title("Survived by Pclas")

# Check out missing values

sns.barplot(x = 0, y = "index", data = df_raw.isnull().sum().reset_index()).set_title("NA in data")


sns.boxplot(x = "Survived",y = "Age", data = df_raw).set_title("Age by target")

sns.boxplot(x = "Pclass",y = "Age",hue = "Survived", data = df_raw).set_title("Age by Pclass and target")

# Average age by pclass

sns.barplot(data = df_raw.groupby("Pclass")["Age"].aggregate("mean").reset_index(),
            x = "Pclass", y = "Age").set_title("Average age by pclass")
