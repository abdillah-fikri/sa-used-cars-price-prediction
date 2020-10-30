---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown//md,scripts//py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
    metadata:
      interpreter:
        hash: 9147bcb9e0785203a659ab3390718fd781c9994811db246717fd6ffdcf1dd807
    name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
---

<!-- #region id="IN1jfOnfZIqT" -->
# Data Importing
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T14:24:16.108907Z", "iopub.status.busy": "2020-10-08T14:24:16.107910Z", "iopub.status.idle": "2020-10-08T14:24:21.655591Z", "shell.execute_reply": "2020-10-08T14:24:21.651602Z", "shell.execute_reply.started": "2020-10-08T14:24:16.108907Z"} id="vBMQYoVFZIqV"
import numpy as np
import pandas as pd
import category_encoders as ce

from utils import null_checker, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
```

```python
df = pd.read_csv('../data/processed/after_prep.csv')
df.head()
```

```python
df.info()
```

<!-- #region id="g1GS1AAUZIt9" -->
# Preprocessing
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T14:24:34.751690Z", "iopub.status.busy": "2020-10-08T14:24:34.751690Z", "iopub.status.idle": "2020-10-08T14:24:34.765663Z", "shell.execute_reply": "2020-10-08T14:24:34.762660Z", "shell.execute_reply.started": "2020-10-08T14:24:34.751690Z"} id="INV8VvOYZItN"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
```

```python id="TYqvFHW1HqFX"
# Drop missing values
df= df.dropna()
null_checker(df)
```

<!-- #region id="yEgVyyNSZIt9" -->
## Train test split
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T14:24:51.747335Z", "iopub.status.busy": "2020-10-08T14:24:51.747335Z", "iopub.status.idle": "2020-10-08T14:24:51.759305Z", "shell.execute_reply": "2020-10-08T14:24:51.757306Z", "shell.execute_reply.started": "2020-10-08T14:24:51.747335Z"} id="nPxFt6bSZIt-" outputId="50d71945-3c1c-4fe9-bb86-b9ae483a319b"
# melakukan train test split di awal untuk mencegah data leakage
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python
# Define category mapping for label encoding
mapping_owner = {
    'First': 1, 
    'Second': 2, 
    'Third': 3, 
    'Fourth & Above': 4
}
mapping_trans = {
    'Manual': 0, 
    'Automatic': 1, 
}

# Encoding train set
X_train["Owner_Type"] = X_train["Owner_Type"].map(mapping_owner)
X_train["Transmission"] = X_train["Transmission"].map(mapping_trans)
# Encoding test set
X_test["Owner_Type"] = X_test["Owner_Type"].map(mapping_owner)
X_test["Transmission"] = X_test["Transmission"].map(mapping_trans)
```

```python execution={"iopub.execute_input": "2020-10-08T14:24:52.214085Z", "iopub.status.busy": "2020-10-08T14:24:52.213088Z", "iopub.status.idle": "2020-10-08T14:24:52.385628Z", "shell.execute_reply": "2020-10-08T14:24:52.384657Z", "shell.execute_reply.started": "2020-10-08T14:24:52.213088Z"} id="kcMLnvJxZIuD"
# One hot encoding for low cardinality feature + Brand
col_to_encode = ['Location', 'Fuel_Type', 'Brand']
oh_encoder = ce.OneHotEncoder(cols=col_to_encode,
                              use_cat_names=True)
oh_encoder.fit(X_train)

# Encoding train set
X_train = oh_encoder.transform(X_train)
# Encoding test set
X_test = oh_encoder.transform(X_test)
```

```python
# Target encoding for high cardinality feature
col_to_encode = X_train.select_dtypes("object").columns
encoder = ce.TargetEncoder(cols=col_to_encode)
encoder.fit(X_train, y_train)

# Encoding train set
X_train = encoder.transform(X_train)
# Encoding test set
X_test = encoder.transform(X_test)
```

<!-- #region id="bw10NXJkIuLs" -->
## Feature Selection
<!-- #endregion -->

```python id="7u-fc0svIuLt"
# Memfilter feature dengan korelasi tinggi
corr_price = X_train.join(y_train).corr()['Price']
index = corr_price[(corr_price < -0.20) | (corr_price > 0.20)].index

X_train =  X_train[index[:-1]]
X_test = X_test[index[:-1]]
```

<!-- #region id="wV2sjkqEZIup" -->
# Modeling
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T15:10:17.976797Z", "iopub.status.busy": "2020-10-08T15:10:17.975799Z", "iopub.status.idle": "2020-10-08T15:10:17.988765Z", "shell.execute_reply": "2020-10-08T15:10:17.987767Z", "shell.execute_reply.started": "2020-10-08T15:10:17.976797Z"} id="Oux2OxeDZIu2"
tree_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()
xgb_model = XGBRegressor()
lgb_model = LGBMRegressor()
cat_model = CatBoostRegressor(silent=True)
lr_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()

models = {'DecisionTree' : tree_model,
          'RandomForest' : rf_model,
          'XGBoost' : xgb_model,
          'CatBoost' : cat_model,
          'LightGBM' : lgb_model,
          'Linear': lr_model,
          'Lasso': lasso_model,
          'Ridge': ridge_model}
```

<!-- #region id="kCSEOF35MoSB" -->
### Unscaled dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 38364, "status": "ok", "timestamp": 1602353945658, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="890d1059-fe50-4ed7-87d9-16413c775534"
# evaluasi model memakai function
unscaled = evaluate_model(models, X_train, X_test, y_train, y_test)
```

<!-- #region id="AodaQJBNMtob" -->
### Scaled dataset
<!-- #endregion -->

```python id="2lQZQbORMwYB"
# Scaling data
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 81010, "status": "ok", "timestamp": 1602353988430, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="90af7df3-a745-4722-d77d-53f144212a91"
# evaluasi model memakai function
scaled = evaluate_model(models, X_train_scaled, X_test_scaled, y_train, y_test)
```

### Summarizing

```python
unscaled['Dataset Version'] = 'dropna + selected + unscaled'
scaled['Dataset Version'] = 'dropna + selected + scaled'
```

```python
dropna_selected = pd.concat([unscaled, scaled], axis=0)
dropna_selected
```

```python
dropna_selected.to_csv('../data/processed/summary_dropna_selected.csv')
```
