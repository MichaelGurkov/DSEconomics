import pandas as pd

def missing_values_table(df):
    mis_val = df.isnull().sum()
    
    mis_val_percent = 100 * mis_val / len(df)
    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    mis_val_table = mis_val_table.rename(
        columns = {0:"missing_values", 1: "missing_values_percent"})
    
    mis_val_table = mis_val_table[mis_val_table["missing_values"] != 0]
    
    mis_val_table = mis_val_table.sort_values("missing_values_percent",
                                              ascending=False).round(1)
    
    print("Your selected  dataframe has " + str(df.shape[1]) + " columns.\n",
          "There are " + str(mis_val_table.shape[0]) + 
          " columns that have missing values")
    
    return mis_val_table

