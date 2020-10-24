---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown//md,scripts//py:percent
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: 'Python 3.8.6 64-bit (''venv-ds'': venv)'
    metadata:
      interpreter:
        hash: 0def509d864da8b5a1806818af2135cd2a80c010b2a4e961690c3f19f43c05fc
    name: 'Python 3.8.6 64-bit (''venv-ds'': venv)'
---

# Exploratory Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import missingno as msno
from utils import *
```

```python
df = pd.read_csv("../data/processed/after_prep.csv")
df.head()
```

```python
df.describe()
```

```python
msno.matrix(df, color=(0, 0.4, 0.7), figsize=(8, 4))
```

```python
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
sns.histplot(data=df, x="Price", ax=ax[0])
sns.boxplot(data=df, x="Price", ax=ax[1])
plt.show()
```

```python
df[df["Price"] > 80]
```

```python
num_cols = [
    col for col in df.drop(columns="Price").columns if df[col].dtype != "object"
]
cat_cols = [
    col for col in df.drop(columns="Price").columns if df[col].dtype == "object"
]
```

```python
plt.figure(figsize=(15, 6))

for index, col in enumerate(num_cols[:3]):
    plt.subplot(2, 3, index + 1)
    sns.histplot(data=df, x=col)

for index, col in enumerate(num_cols[:3]):
    plt.subplot(2, 3, index + 4)
    sns.boxplot(data=df, x=col)

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(15, 6))

for index, col in enumerate(num_cols[3:]):
    plt.subplot(2, 3, index + 1)
    sns.histplot(data=df, x=col)

for index, col in enumerate(num_cols[3:]):
    plt.subplot(2, 3, index + 4)
    sns.boxplot(data=df, x=col)

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(8, 6))
plt.subplot(211)
sns.histplot(data=df, x="Kilometers_Driven")
plt.subplot(212)
sns.boxplot(data=df, x="Kilometers_Driven")
plt.tight_layout()
plt.show()
```

```python
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
```

```python
plt.figure(figsize=(8, 6))
plt.subplot(211)
sns.histplot(data=df, x="Kilometers_Driven")
plt.subplot(212)
sns.boxplot(data=df, x="Kilometers_Driven")
plt.tight_layout()
plt.show()
```

```python
df[df.Seats >= 9]
```

```python
df.describe(include=["object"])
```

```python
cat_cols
```

```python
cols_toplot = ["Fuel_Type", "Transmission", "Location", "Owner_Type"]
plt.figure(figsize=(12, 8))
countplot_annot(2, 2, data=df, columns=cols_toplot, rotate=45, rcol=cols_toplot)
plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(15, 4))
countplot_annot(1, 1, data=df, columns=["Brand"], rotate=90, rcol=["Brand"])
plt.ylim(0, 1300)
plt.show()
```

```python
fig = px.bar(
    y=df["Series"].value_counts()[:50],
    x=df["Series"].value_counts()[:50].keys(),
    text=df["Series"].value_counts()[:50],
)
fig.update_layout(autosize=False, width=1200, height=500)
fig.show()
```

```python
plt.figure(figsize=(15, 4))

for index, col in enumerate(["Fuel_Type", "Transmission", "Owner_Type"]):
    plt.subplot(1, 3, index + 1)
    sns.barplot(data=df, x=col, y="Price")

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(15, 4))

for index, col in enumerate(["Fuel_Type", "Transmission", "Owner_Type"]):
    plt.subplot(1, 3, index + 1)
    sns.boxplot(data=df, x=col, y="Price")

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Location", y="Price")
plt.show()
```

```python
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Location", y="Price")
plt.show()
```

```python
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x="Brand", y="Price")
plt.xticks(rotation=90)
plt.show()
```

```python
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Brand", y="Price")
plt.xticks(rotation=90)
plt.show()
```

```python
import category_encoders as ce

t_encoder = ce.TargetEncoder()
df_temp = t_encoder.fit_transform(df.drop(columns=["Price"]), df["Price"])
df_temp = pd.concat([df_temp, df["Price"]], axis=1)

plt.figure(figsize=(14, 10))
sns.heatmap(df_temp.corr(), annot=True, linewidths=0.5, fmt=".2f")
plt.show()
```

```python
import plotly.express as px

fig = px.scatter(
    df,
    x="Power (bhp)",
    y="Engine (CC)",
    size="Price",
    color="Transmission",
    hover_name="Brand",
    log_x=True,
    size_max=25,
)

fig.update_layout(title="Engine and Power correlation")
fig.show()
```

```python
fig = px.scatter(
    df,
    x="Mileage (kmpl)",
    y="Engine (CC)",
    size="Price",
    color="Fuel_Type",
    hover_name="Brand",
    log_x=True,
    size_max=25,
)

fig.update_layout(title="Engine and Mileage correlation")
fig.show()
```

```python
from sklearn.preprocessing import MinMaxScaler

df_grp = df.groupby("Year")["Price", "Mileage (kmpl)"].mean()
df_grp_scaled = MinMaxScaler().fit_transform(df_grp)
df_grp_scaled = pd.DataFrame(df_grp_scaled, columns=df_grp.columns, index=df_grp.index)

trace1 = go.Scatter(
    x=df_grp_scaled.index, y=df_grp_scaled["Price"], mode="lines+markers", name="Price"
)

trace2 = go.Scatter(
    x=df_grp_scaled.index,
    y=df_grp_scaled["Mileage (kmpl)"],
    mode="lines",
    name="Mileage (kmpl)",
)

data = [trace1, trace2]
layout = go.Layout(title="Price and Mileage over the time", xaxis=dict(title="Year"))

fig = go.Figure(data=data, layout=layout)


fig.show()
```

```python
import plotly.graph_objects as go

df_grp = df.groupby(["Brand", "Transmission"], as_index=False)["Price"].median()
df_grp.sort_values(by="Price", inplace=True)
df_grp.head()

fig = px.bar(
    df_grp,
    x="Brand",
    y="Price",
    color="Transmission",
    title="Median price by Brand",
    height=500,
    width=800,
)

fig.show()
```

```python
df_grp = df.groupby(["Brand"], as_index=False).agg(
    Median_Price=("Price", "median"), Count=("Price", "count")
)
df_grp.sort_values(by="Median_Price", inplace=True)
df_grp.head()

fig = px.bar(
    df_grp,
    x="Brand",
    y="Count",
    title="Count of Cars by Brand",
    text="Count",
    height=500,
    width=800,
)

fig.add_trace(
    go.Scatter(x=df_grp["Brand"], y=df_grp["Median_Price"] * 10, name="Median Price")
)

fig.show()
```

# Update for Final Presentation

```python
df.head()
```

```python
from sklearn.preprocessing import MinMaxScaler

df_grp = df.groupby("Brand", as_index=False).agg(
    car_count=("Brand", "count"), car_price=("Price", "median")
)
df_grp.sort_values(by="car_price", ascending=False, inplace=True)

scaler = MinMaxScaler()
df_grp["car_count_scaled"] = (
    scaler.fit_transform(df_grp["car_count"].values.reshape(-1, 1)) + 0.02
)
df_grp["car_price_scaled"] = (
    scaler.fit_transform(df_grp["car_price"].values.reshape(-1, 1)) + 0.02
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=df_grp["Brand"],
        y=df_grp["car_count_scaled"],
        text=df_grp["car_count"],
        textposition="auto",
        name="CAR COUNT",
    )
)

fig.add_trace(
    go.Bar(
        x=df_grp["Brand"],
        y=df_grp["car_price_scaled"],
        text=df_grp["car_price"],
        textposition="auto",
        name="MEDIAN CAR PRICE",
    )
)
fig.update_layout(height=500, width=1400)

fig.show()
```

```python
from sklearn.preprocessing import MinMaxScaler

df_grp = df.groupby("Brand", as_index=False).agg(
    car_count=("Brand", "count"), car_price=("Price", "median")
)
df_grp.sort_values(by="car_price", ascending=False, inplace=True)
df_grp = df_grp[df_grp["car_count"] > 5]

scaler = MinMaxScaler()
df_grp["car_count_scaled"] = (
    scaler.fit_transform(df_grp["car_count"].values.reshape(-1, 1)) + 0.02
)
df_grp["car_price_scaled"] = (
    scaler.fit_transform(df_grp["car_price"].values.reshape(-1, 1)) + 0.02
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=df_grp["Brand"],
        y=df_grp["car_count_scaled"],
        text=df_grp["car_count"],
        textposition="auto",
        name="CAR COUNT",
    )
)

fig.add_trace(
    go.Bar(
        x=df_grp["Brand"],
        y=df_grp["car_price_scaled"],
        text=df_grp["car_price"],
        textposition="auto",
        name="MEDIAN CAR PRICE",
    )
)
fig.update_layout(height=500, width=1400)

fig.show()
```

```python
df_grp = df.groupby("Brand", as_index=False).agg(
    car_count=("Brand", "count"), car_price=("Price", "median")
)
df_grp[df_grp["car_count"] < 5]
```

```python
from sklearn.preprocessing import MinMaxScaler

df_grp = df.groupby("Location", as_index=False).agg(
    car_count=("Location", "count"), car_price=("Price", "median")
)
df_grp.sort_values(by="car_price", ascending=False, inplace=True)

scaler = MinMaxScaler()
df_grp["car_count_scaled"] = (
    scaler.fit_transform(df_grp["car_count"].values.reshape(-1, 1)) + 0.1
)
df_grp["car_price_scaled"] = (
    scaler.fit_transform(df_grp["car_price"].values.reshape(-1, 1)) + 0.1
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=df_grp["Location"],
        y=df_grp["car_count_scaled"],
        text=df_grp["car_count"],
        textposition="auto",
        name="CAR COUNT",
    )
)

fig.add_trace(
    go.Bar(
        x=df_grp["Location"],
        y=df_grp["car_price_scaled"],
        text=df_grp["car_price"],
        textposition="auto",
        name="MEDIAN CAR PRICE",
    )
)
fig.update_layout(height=400, width=1000)

fig.show()
```

```python
df.head()
```

```python
df_grp = df.groupby(["Owner_Type", "Brand"]).agg(
    car_count=("Owner_Type", "count"), median_car_price=("Price", "median")
)
df_grp.sort_values(by="car_count", ascending=False, inplace=True)
df_grp.head(20)
```

```python
df_grp = df.groupby(["Owner_Type", "Location", "Brand"], as_index=False).agg(
    car_count=("Owner_Type", "count"), median_car_price=("Price", "median")
)
df_grp.sort_values(by="car_count", ascending=False, inplace=True)
df_grp[df_grp["Location"] == "Ahmedabad"].head(20)
```

```python
df_grp = df.groupby(["Fuel_Type", "Brand"]).agg(
    car_count=("Fuel_Type", "count"), car_price=("Price", "median")
)
df_grp.sort_values(by="car_count", ascending=False, inplace=True)
df_grp.head(20)
```

```python
df_grp = df.groupby(["Fuel_Type", "Location", "Brand"], as_index=False).agg(
    car_count=("Fuel_Type", "count"), median_car_price=("Price", "median")
)
df_grp.sort_values(by="car_count", ascending=False, inplace=True)
df_grp[df_grp["Location"] == "Ahmedabad"].head(20)
```

```python
df.head()
```

```python
df.describe()
```

```python
def segment(price):
    if price <= 5.64:
        return "Low"
    elif price <= 20:
        return "Middle"
    else:
        return "High"

df["Segment"] = df["Price"].apply(segment)

fig = px.histogram(df, x="Segment")
fig.show()
```

```python
fig = px.histogram(df, x="Segment", color="Owner_Type", barnorm="percent")
fig.update_layout(yaxis=dict(title="Proportion"))
fig.show()
```

```python
fig = px.histogram(df, x="Segment", color="Location", barnorm="percent")
fig.update_layout(yaxis=dict(title="Proportion"))
fig.show()
```

```python
fig = px.histogram(df, x="Segment", color="Fuel_Type", barnorm="percent")
fig.update_layout(yaxis=dict(title="Proportion"))
fig.show()
```

```python

```
