
import sys

sys.path.append("G:\\My Drive\\MTA\\OnlineCourse\\python_scripts")

# from auxilary_functions import missing_values_table  

import import_and_process

from import_and_process import train_data

from import_and_process import test_data

# Import libraries
#-----------------------------------------------------------------------------
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import os

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns



# Categorical variables encoding

le = LabelEncoder()

le_count = 0

ohe = OneHotEncoder()

ohe_count = 0

for temp_col in train_data.columns.values:
    if train_data[temp_col].dtype == "object":
        print(temp_col)
        if train_data[temp_col].nunique() > 2:
            ohe.fit(train_data[temp_col].values[Ellipsis, None])
            train_temp_encoding = pd.DataFrame(ohe.transform(train_data[temp_col].values[Ellipsis, None]).toarray())
            train_temp_encoding.columns = ohe.get_feature_names()
            train_data.drop(columns = [temp_col], inplace=True)
            train_data = pd.concat([train_data, train_temp_encoding],axis=1)
            
            test_temp_encoding = pd.DataFrame(ohe.transform(test_data[temp_col].values[Ellipsis, None]).toarray())
            test_temp_encoding.columns = ohe.get_feature_names()
            test_data.drop(columns = [temp_col], inplace=True)
            test_data = pd.concat([test_data, test_temp_encoding],axis=1)
            
            # test_data[temp_col] = ohe.transform(test_data[temp_col].values[Ellipsis, None])
            ohe_count = ohe_count + 1
        else:
            le.fit(train_data[temp_col].values[Ellipsis, None])
            train_data[temp_col] = le.transform(train_data[temp_col].values[Ellipsis, None])
            test_data[temp_col] = le.transform(test_data[temp_col].values[Ellipsis, None])
            le_count = le_count + 1

print(str(le_count) + " columns were label encoded")
print(str(ohe_count) + " columns were one hot encoded")



