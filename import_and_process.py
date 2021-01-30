
# Import libraries
#-----------------------------------------------------------------------------
import numpy as np

import pandas as pd


import warnings
warnings.filterwarnings("ignore")

# Import data
#-----------------------------------------------------------------------------

train_data = pd.read_csv("C:\\Users\\micha\\Documents\\data\\application_train.csv")

test_data = pd.read_csv("C:\\Users\\micha\\Documents\\data\\application_test.csv")

# missing_df = train_data.isnull().sum().sort_values(ascending=False) / train_data.shape[0]

# missing_df.nlargest(25).sort_values(ascending = True).plot(kind = "barh")

# Impute NA's with mode

train_data.fillna(train_data.mode().iloc[0], inplace=True)

test_data.fillna(test_data.mode().iloc[0], inplace=True)

print("import and process done")
