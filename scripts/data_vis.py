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
#         hash: 9147bcb9e0785203a659ab3390718fd781c9994811db246717fd6ffdcf1dd807
#     name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
# ---

# %% [markdown]
# # Exploratory Data Analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import missingno as msno
from utils import *

# %%
# Import data
df = pd.read_csv("../data/processed/after_prep.csv")
df.head()

# %%
# Summary statistic
df.describe()

# %%
# Missing value matrix
msno.matrix(df, color=(0, 0.4, 0.7), figsize=(15, 6))

# %% [markdown]
# ## Univariate Analysis

# %%
# Plot target variable distribution
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
sns.histplot(data=df, x="Price", ax=ax[0])
sns.boxplot(data=df, x="Price", ax=ax[1])
plt.show()

# %%
# Examine price greater than 80
df[df["Price"] > 80]

# %%
# Categorize cols variables for plotting
num_cols = [
    col for col in df.drop(columns="Price").columns if df[col].dtype != "object"
]
cat_cols = [
    col for col in df.drop(columns="Price").columns if df[col].dtype == "object"
]

# %% [markdown]
# ### Numerical features

# %%
# Plotting first 3 numerical features distribution
plt.figure(figsize=(15, 6))

for index, col in enumerate(num_cols[:3]):
    plt.subplot(2, 3, index + 1)
    sns.histplot(data=df, x=col)

for index, col in enumerate(num_cols[:3]):
    plt.subplot(2, 3, index + 4)
    sns.boxplot(data=df, x=col)

plt.tight_layout()
plt.show()

# %%
# Plot last 3 numerical features distribution
plt.figure(figsize=(15, 6))

for index, col in enumerate(num_cols[3:]):
    plt.subplot(2, 3, index + 1)
    sns.histplot(data=df, x=col)

for index, col in enumerate(num_cols[3:]):
    plt.subplot(2, 3, index + 4)
    sns.boxplot(data=df, x=col)

plt.tight_layout()
plt.show()

# %%
# Examine outlier on Kilometers Driven feature
plt.figure(figsize=(8, 6))
plt.subplot(211)
sns.histplot(data=df, x="Kilometers_Driven")
plt.subplot(212)
sns.boxplot(data=df, x="Kilometers_Driven")
plt.tight_layout()
plt.show()

# %%
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape

# %%
# Plot after outlier removed
plt.figure(figsize=(8, 6))
plt.subplot(211)
sns.histplot(data=df, x="Kilometers_Driven")
plt.subplot(212)
sns.boxplot(data=df, x="Kilometers_Driven")
plt.tight_layout()
plt.show()

# %%
# Examine car that have seats more than or equal 9
df[df.Seats >= 9]

# %% [markdown]
# ### Categorical features

# %%
df.describe(include=["object"])

# %%
# Make count plot the categorical features
cols_toplot = ["Fuel_Type", "Transmission", "Location", "Owner_Type"]
plt.figure(figsize=(12, 8))
countplot_annot(2, 2, data=df, columns=cols_toplot, rotate=45, rcol=cols_toplot)
plt.tight_layout()
plt.show()

# %%
# Count plot for Brand feature
plt.figure(figsize=(15, 4))
countplot_annot(1, 1, data=df, columns=["Brand"], rotate=90, rcol=["Brand"])
plt.ylim(0, 1300)
plt.show()

# %% [markdown]
# ## Bivariate Analysis

# %%
# Bar plot with mean value of Price by Fuel_Type, Transmission, and Owner_Type
plt.figure(figsize=(15, 4))

for index, col in enumerate(["Fuel_Type", "Transmission", "Owner_Type"]):
    plt.subplot(1, 3, index + 1)
    sns.barplot(data=df, x=col, y="Price")

plt.tight_layout()
plt.show()

# %%
# Box plot with mean value of Price by Fuel_Type, Transmission, and Owner_Type
plt.figure(figsize=(15, 4))

for index, col in enumerate(["Fuel_Type", "Transmission", "Owner_Type"]):
    plt.subplot(1, 3, index + 1)
    sns.boxplot(data=df, x=col, y="Price")

plt.tight_layout()
plt.show()

# %%
# Bar plot with mean value of Price by Location
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Location", y="Price")
plt.show()

# %%
# Box plot with mean value of Price by Location
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Location", y="Price")
plt.show()

# %%
# Bar plot with mean value of Price by Brand
plt.figure(figsize=(15, 8))
sns.barplot(data=df, x="Brand", y="Price")
plt.xticks(rotation=90)
plt.show()

# %%
# Box plot with mean value of Price by Brand
plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Brand", y="Price")
plt.xticks(rotation=90)
plt.show()

# %%
# Correlation heatmap
import category_encoders as ce

t_encoder = ce.TargetEncoder()
df_temp = t_encoder.fit_transform(df.drop(columns=["Price"]), df["Price"])
df_temp = pd.concat([df_temp, df["Price"]], axis=1)
df_temp.drop(["Segment", "Zone"], axis=1, inplace=True)

sns.set(font_scale=1.8)
plt.figure(figsize=(14, 10))
sns.heatmap(df_temp.corr(), annot=True, linewidths=0.5, fmt=".2f", annot_kws={"size": 14})
plt.show()

# %%
# Bubble plot Engine and Power correlation with Price and Transmission
import plotly.express as px

fig = px.scatter(
    df,
    x="Power (bhp)",
    y="Engine (CC)",
    size="Price",
    color="Transmission",
    hover_name="Brand",
    log_x=False,
    size_max=30,
    height=600,
    width=900
)

fig.update_layout(title="Engine and Power correlation")
fig.show()

# %%
# Bubble plot Engine and Mileage correlation with Price and Transmission
fig = px.scatter(
    df,
    x="Mileage (kmpl)",
    y="Engine (CC)",
    size="Price",
    color="Fuel_Type",
    hover_name="Brand",
    log_x=True,
    size_max=25,
    category_orders={"Fuel_Type": ["Diesel", "Petrol", "CNG", "LPG", "Electric"]},
)

fig.update_layout(
    title=dict(text="Engine and Mileage correlation"),
)
fig.show()

# %%
# Plot time series of Price and Mileage
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

# %%
# Bar plot with median price by Brand
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

# %%
# Bar plot with count of car vs Median price
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

# %% [markdown]
# # Update for Final Presentation

# %%
# pio.templates.default = "presentation"
# pio.templates

# %%
df.head()

# %%
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

# %%
# Plot car count vs median price comparison
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
fig.update_layout(
    title="Car count vs median price comparison",
    height=400, width=1000
)
fig.show()

# %%
# Excluding car brand with lower than 5 count
df_grp = df.groupby("Brand", as_index=False).agg(
    car_count=("Brand", "count"), car_price=("Price", "median")
)
df_grp[df_grp["car_count"] < 5]

# %%
# Plot car count vs median price comparison excluding car brand with lower than 5 count
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
fig.update_layout(
    title="Car count vs median price comparison",
    height=400, width=1000
)
fig.show()

# %% [markdown]
# ## Market Analysis

# %%
pio.templates.default = "presentation"
pio.templates

# %%
market_value = pd.read_csv("../data/raw/market_value.csv")
market_value

# %%
fig = px.bar(
    market_value, x="Year", y="Market Value",
    color="Market Value", color_continuous_scale=px.colors.sequential.Plasma,
    title="USED CAR MARKET<br>(Revenue in USD billion, India, 2019-2025)"
)
fig.update_layout(
    xaxis=None,
    height=500*1.2,
    width=700*1.2
)
fig.show()


# %%
# Make and plot price segmentation
def segment(price):
    if price <= 5.64:
        return "Low"
    elif price <= 20:
        return "Middle"
    else:
        return "High"


df["Segment"] = df["Price"].apply(segment)

fig = px.histogram(df, x="Segment", y="Price", histfunc="avg")
fig.show()


# %%
# Pie plot for distributon of market segmentation
df_segment = df.groupby(["Segment", "Brand"], as_index=False).agg(Count=("Price", "count"))
df_segment.sort_values(by="Count", ascending=False, inplace=True)

fig = px.pie(df_segment, values="Count", names="Segment")
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(
    title="Market Segmentation",
    showlegend=False, font_size=18
)
fig.show()

# %%
df_segment

# %%
# Pie plot for Car brand on Low segmentation
df_low = df_segment[df_segment["Segment"] == "Low"]
df_low["Brand_2"] = df_low.apply(
    lambda data: "Other" if data["Count"] < 70 else data["Brand"], axis=1
)
fig = px.pie(df_low, values="Count", names="Brand_2")
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(title="Low", showlegend=False, font_size=16)
fig.show()

# %%
# Pie plot for Car brand on Middle segmentation
df_middle = df_segment[df_segment["Segment"] == "Middle"]
df_middle["Brand_2"] = df_middle.apply(
    lambda data: "Other" if data["Count"] < 35 else data["Brand"], axis=1
)
fig = px.pie(df_middle, values="Count", names="Brand_2")
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(title="Middle", showlegend=False, font_size=16)
fig.show()

# %%
# Pie plot for Car brand on High segmentation
df_high = df_segment[df_segment["Segment"] == "High"]
df_high["Brand_2"] = df_high.apply(
    lambda data: "Other" if data["Count"] < 15 else data["Brand"], axis=1
)
fig = px.pie(df_high, values="Count", names="Brand_2")
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(title="High", showlegend=False, font_size=16)
fig.show()

# %%
# Bar plot for distributon of market segmentation
df_grp = df.groupby("Segment", as_index=False).agg(Count=("Brand", "count"))
df_grp.sort_values(by="Count", ascending=False, inplace=True)
fig = go.Figure(
    data=[
        go.Bar(
            x=df_grp["Segment"],
            y=df_grp["Count"],
            text=df_grp["Count"],
            textposition="auto"
        )
    ]
)
fig.update_layout(
    title="Market Segments",
    xaxis=dict(title=""),
    yaxis=dict(title="Count"),
    height=500,
    width=800,
    margin=dict(l=100, r=50, t=100, b=50),
)
fig.show()


# %%
fig = px.colors.qualitative.swatches()
fig.show()

# %%
# Proprotion bar plot for Market segmentation based on Transmission
fig = px.histogram(
    df,
    x="Segment",
    color="Transmission",
    barnorm="percent",
)
fig.update_layout(
    title="Market segmentation based on Transmission",
    xaxis=None,
    yaxis=None,
    height=500,
    width=700,
    margin=dict(l=100, r=100, t=100, b=50),
)
fig.show()

# %%
# Proprotion bar plot for Market segmentation based on Owner Type
fig = px.histogram(
    df,
    x="Segment",
    color="Owner_Type",
    barnorm="percent",
    category_orders={"Owner_Type": ["First", "Second", "Third", "Fourth & Above"]}
)
fig.update_layout(
    title="Market segmentation based on Owner Type",
    xaxis=dict(title=""),
    yaxis=dict(title="Proportion"),
    height=500,
    width=700,
    margin=dict(l=100, r=100, t=100, b=50),
)
fig.show()


# %%
# Proprotion bar plot for Market segmentation based on Zone
def zone(data):
    if data in ["Kolkata"]:
        return "Eastern"
    elif data in ["Delhi", "Jaipur"]:
        return "Northern"
    elif data in ["Ahmedabad", "Mumbai", "Pune"]:
        return "Western"
    else:
        return "Southern"


# Northern ["Delhi", "Jaipur"]
# Central ["Kolkata"]
# Western ["Ahmedabad", "Mumbai", "Pune"]
# Southern ["Chennai", "Hyderabad", "Bengaluru", "Coimbatore", "Kochi"]

df["Zone"] = df["Location"].apply(zone)

fig = px.histogram(df, x="Segment", color="Zone", barnorm="percent")
fig.update_layout(
    title="Market segmentation based on Zone",
    xaxis=None,
    yaxis=None,
    height=500,
    width=700,
    margin=dict(l=100, r=100, t=100, b=50),
)
fig.show()

# %%
# Proprotion bar plot for Market segmentation based on Fuel Type
fig = px.histogram(
    df,
    x="Segment",
    color="Fuel_Type",
    barnorm="percent",
    category_orders={
        "Fuel_Type": ["Diesel", "Petrol", "CNG", "LPG", "Electric"],
        "Segment": ["Low", "Middle", "High"],
    },
)
fig.update_layout(
    title="Market segmentation based on Fuel Type",
    xaxis=dict(title=""),
    yaxis=dict(title="Proportion"),
    height=500,
    width=700,
    margin=dict(l=100, r=100, t=100, b=50),
)
fig.show()

# %% [markdown]
#
