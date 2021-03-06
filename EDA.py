
import sys

sys.path.append("C:\\Users\\micha\\Documents\\DSEconomis")

# from auxilary_functions import missing_values_table  

import import_and_process

import preprocess_and_clean

from preprocess_and_clean import train_data

from preprocess_and_clean import test_data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Anomalies

(train_data["DAYS_EMPLOYED"] * (-1)).describe()

train_data["DAYS_EMPLOYED"] .plot(kind = "hist",
                                  title="Days of employment before loan")

train_data = train_data[train_data["DAYS_EMPLOYED"].abs() < (365 * 65)]

# Correlations

corr = train_data.corr()["TARGET"].abs().sort_values(ascending=False)

corr[1:15].plot(kind = "barh")

## Age relationship

plt.hist(train_data["DAYS_BIRTH"] / (-365), )

sns.kdeplot(train_data.loc[train_data["TARGET"] == 0,"DAYS_BIRTH"] / (-365),
            label="good customer")

sns.kdeplot(train_data.loc[train_data["TARGET"] == 1,"DAYS_BIRTH"] / (-365),
            label="bad customer")

plt.legend()

plt.show()

# Cutting the age to categories

age_data = train_data[["TARGET","DAYS_BIRTH"]]

age_data["YEARS"] = age_data["DAYS_BIRTH"] /(-365)

age_data["YEARS_BINNED"] = pd.cut(age_data["YEARS"],
                                  bins= np.linspace(20,70,11))

age_groups = age_data.groupby("YEARS_BINNED").mean()

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])


