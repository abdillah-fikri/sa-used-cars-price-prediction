# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
#     metadata:
#       interpreter:
#         hash: 8d4d772f21767a3a72f3356b4ab1badff3b831eb21eba306d4ebdf1fe7777d12
#     name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
# ---

# %%
# Import library
import numpy as np
import pandas as pd

from utils import null_checker

# %%
# Import data
df = pd.read_csv("../data/raw/used_car_data.csv")
print("Shape:", df.shape)
df.head()

# %%
# check info
df.info()

# %%
# check null
null_checker(df)

# %%
df['Brand'] = df['Name'].apply(lambda x: x.split(' ')[0])
df['Series'] = df['Name'].apply(lambda x: x.split(' ')[1])
df['Type'] = df['Name'].apply(lambda x: x.split(' ')[2])
df.drop(columns='Name', inplace=True)

# %%
# Check features unit
print(
    "Satuan pada feature Mileage:",
    df["Mileage"].apply(lambda x: x if pd.isna(x) else x.split(" ")[1]).unique(),
)
print(
    "Satuan pada feature Engine:",
    df["Engine"].apply(lambda x: x if pd.isna(x) else x.split(" ")[1]).unique(),
)
print(
    "Satuan pada feature Power:",
    df["Power"].apply(lambda x: x if pd.isna(x) else x.split(" ")[1]).unique(),
)

# %%
# Check invalid value
print(
    "Invalid Value pada feature Mileage:",
    pd.Series([x for x in df["Mileage"] if str(x).split(" ")[0].isalpha()]).unique(),
)
print(
    "Invalid Value pada feature Engine:",
    pd.Series([x for x in df["Engine"] if str(x).split(" ")[0].isalpha()]).unique(),
)
print(
    "Invalid Value pada feature Power:",
    pd.Series([x for x in df["Power"] if str(x).split(" ")[0].isalpha()]).unique(),
)

# %%
# Remove features unit and convert to numeric
df["Mileage (kmpl)"] = df["Mileage"].apply(lambda x: x if pd.isna(x) else x.split(" ")[0])
df["Engine (CC)"] = df["Engine"].apply(lambda x: x if pd.isna(x) else x.split(" ")[0])
df["Power (bhp)"] = df["Power"].apply(lambda x: x if pd.isna(x) else x.split(" ")[0])

df["Mileage (kmpl)"] = pd.to_numeric(df["Mileage (kmpl)"], errors="coerce")
df["Engine (CC)"] = pd.to_numeric(df["Engine (CC)"], errors="coerce")
df["Power (bhp)"] = pd.to_numeric(df["Power (bhp)"], errors="coerce")

df.drop(columns=["Mileage", "Engine", "Power"], inplace=True)

# %%
# Check result
df.head()

# %%
# Summary statistic
df.describe()

# %%
# Check milage 0
df[df["Mileage (kmpl)"] == 0]

# %%
# Seats 0 value
df[df["Seats"] == 0]

# %%
# Replace 0 value to nan
df["Mileage (kmpl)"] = df["Mileage (kmpl)"].replace(0, np.nan)
df["Seats"] = df["Seats"].replace(0, np.nan)

# %%
# Check unique value
cat_cols = [col for col in df.columns if df[col].dtypes == "object"]
df[cat_cols].nunique()

# %%
for col in cat_cols:
    print(col, df[col].unique(), "\n")

# %%
# Replace duplicated value
df["Brand"] = df["Brand"].replace("ISUZU", "Isuzu")

# %%
df.head()

# %%
df.describe()

# %%
df.describe(include=["object"])

# %%
null_checker(df)

# %%
# Replace Electric car's mileage to 0
df.loc[df['Fuel_Type']=='Electric', 'Mileage (kmpl)'] = df.loc[df['Fuel_Type']=='Electric', 'Mileage (kmpl)'].replace(np.nan, 0)
df.loc[df['Fuel_Type']=='Electric']

# %%
df.to_csv("../data/processed/after_prep.csv", index=False)
