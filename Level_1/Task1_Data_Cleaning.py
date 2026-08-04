import calendar
import re
import pandas as pd
df = pd.read_csv("Level_1/data.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df = df.fillna('Unknown')
df = df.drop_duplicates()
df["Timestamp"]=pd.to_datetime(df["Timestamp"])
df["Month"] =df["Month"].apply(lambda x: calendar.month_name[int(x)])
df["Hour"] = df["Hour"].astype(str)+"hours"
df.to_csv("Level_1/cleaned_data.csv",index=False)
print("Data cleaning completed sucessfully")


