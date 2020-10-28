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
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="IN1jfOnfZIqT" -->
# Data Importing
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T14:24:16.108907Z", "iopub.status.busy": "2020-10-08T14:24:16.107910Z", "iopub.status.idle": "2020-10-08T14:24:21.655591Z", "shell.execute_reply": "2020-10-08T14:24:21.651602Z", "shell.execute_reply.started": "2020-10-08T14:24:16.108907Z"} id="vBMQYoVFZIqV"
import pandas as pd
import numpy as np
import category_encoders as ce
import miceforest as mf
import optuna
import lightgbm as lgb
import xgboost as xgb

from utils import *
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics
from sklearn.impute import SimpleImputer
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
# melakukan train test split di awal untuk mencegah data bocor ke test set saat dilakukan encoding/imputation
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python
# encodes = ['Location','Fuel_Type','Transmission','Owner_Type', 'Brand']
# encoder = ce.OneHotEncoder(cols=encodes,
#                           use_cat_names=True)
# encoder.fit(X_train)

# # encoding train set
# X_train = encoder.transform(X_train)

# # encoding test set
# X_test = encoder.transform(X_test)
```

```python execution={"iopub.execute_input": "2020-10-08T14:24:52.214085Z", "iopub.status.busy": "2020-10-08T14:24:52.213088Z", "iopub.status.idle": "2020-10-08T14:24:52.385628Z", "shell.execute_reply": "2020-10-08T14:24:52.384657Z", "shell.execute_reply.started": "2020-10-08T14:24:52.213088Z"} id="kcMLnvJxZIuD"
# Target encoding
col_to_encode = ['Series', 'Type', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
target_encoder = ce.TargetEncoder(cols=col_to_encode)
target_encoder.fit(X_train, y_train)

# Encoding train set
X_train = target_encoder.transform(X_train)
# Encoding test set
X_test = target_encoder.transform(X_test)
```

<!-- #region id="bw10NXJkIuLs" -->
## Feature Selection
<!-- #endregion -->

```python id="7u-fc0svIuLt"
# Memfilter feature dengan korelasi tinggi
corr_price = X_train.join(y_train).corr()['Price']
index = corr_price[(corr_price < -0.20) | (corr_price > 0.20)].index

X_train_selected = X_train[index[:-1]]
X_test_selected = X_test[index[:-1]]
```

```python
X_train.shape
```

```python
X_train_selected.shape
```

<!-- #region id="wV2sjkqEZIup" -->
# Modeling
<!-- #endregion -->

<!-- #region id="4g_nWqotKl6_" -->
## Functions
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T14:26:01.181734Z", "iopub.status.busy": "2020-10-08T14:26:01.181734Z", "iopub.status.idle": "2020-10-08T14:26:01.202651Z", "shell.execute_reply": "2020-10-08T14:26:01.201684Z", "shell.execute_reply.started": "2020-10-08T14:26:01.181734Z"} id="Qp4QHIuFZIuq"
def get_cv_score(models, X_train, y_train):
    
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    summary = []
    for label, model in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=cv, 
                                    scoring=['r2',
                                             'neg_root_mean_squared_error',
                                             'neg_mean_absolute_error'])
        
        temp = pd.DataFrame(cv_results).copy()
        temp['Model'] = label
        summary.append(temp)
    
    summary = pd.concat(summary)
    summary = summary.groupby('Model').mean()
    
    summary.drop(columns=['fit_time', 'score_time'], inplace=True)
    summary.columns = ['CV R2', 'CV RMSE', 'CV MAE']
    summary[['CV RMSE', 'CV MAE']] = summary[['CV RMSE', 'CV MAE']] * -1
    
    return summary
```

```python execution={"iopub.execute_input": "2020-10-08T14:26:01.236560Z", "iopub.status.busy": "2020-10-08T14:26:01.235563Z", "iopub.status.idle": "2020-10-08T14:26:01.249526Z", "shell.execute_reply": "2020-10-08T14:26:01.248529Z", "shell.execute_reply.started": "2020-10-08T14:26:01.236560Z"} id="BXEr8F5VZIu0"
def evaluate_model(models, X_train, X_test, y_train, y_test):

    summary = {'Model':[], 'Train R2':[], 'Train RMSE':[], 'Train MAE':[],
               'Test R2':[], 'Test RMSE':[], 'Test MAE':[]}

    for label, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        summary['Model'].append(label)

        summary['Train R2'].append(
            metrics.r2_score(y_train, y_train_pred))
        summary['Train RMSE'].append(
            np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
        summary['Train MAE'].append(
            metrics.mean_absolute_error(y_train, y_train_pred))

        summary['Test R2'].append(
            metrics.r2_score(y_test, y_test_pred))
        summary['Test RMSE'].append(
            np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
        summary['Test MAE'].append(
            metrics.mean_absolute_error(y_test, y_test_pred))
    
    summary = pd.DataFrame(summary)
    summary.set_index('Model', inplace=True)

    cv_scores = get_cv_score(models, X_train, y_train)
    summary = summary.join(cv_scores)
    summary = summary[['Train R2', 'CV R2', 'Test R2',
                       'Train RMSE', 'CV RMSE', 'Test RMSE',
                       'Train MAE', 'CV MAE', 'Test MAE']]
    
    return round(summary.sort_values(by='Test RMSE'), 4)
```

<!-- #region id="aR4Sp3UCZIu2" -->
## Base Model
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-08T15:10:17.976797Z", "iopub.status.busy": "2020-10-08T15:10:17.975799Z", "iopub.status.idle": "2020-10-08T15:10:17.988765Z", "shell.execute_reply": "2020-10-08T15:10:17.987767Z", "shell.execute_reply.started": "2020-10-08T15:10:17.976797Z"} id="Oux2OxeDZIu2"
tree_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()
xgb_model = XGBRegressor(objective='reg:squarederror')
lgb_model = LGBMRegressor()
cat_model = CatBoostRegressor(silent=True)
lr_model = LinearRegression()
lasso_model = Lasso()

models = {'DecisionTreeRegressor' : tree_model,
          'RandomForestRegressor' : rf_model,
          'XGBRegressor' : xgb_model,
          'CatBoostRegressor' : cat_model,
          'LGBMRegressor' : lgb_model,
          'LinearRegression': lr_model,
          'LassoRegression': lasso_model}
```

<!-- #region id="kCSEOF35MoSB" -->
### Unscaled dataset
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 38364, "status": "ok", "timestamp": 1602353945658, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="890d1059-fe50-4ed7-87d9-16413c775534"
# evaluasi model memakai function
unscaled = evaluate_model(models, X_train_selected, X_test_selected, y_train, y_test)
```

<!-- #region id="AodaQJBNMtob" -->
### Scaled dataset
<!-- #endregion -->

```python id="2lQZQbORMwYB"
# Scaling data with RobustScaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train_selected)
X_train_selected_scaled_r = scaler.transform(X_train_selected)
X_test_selected_scaled_r = scaler.transform(X_test_selected)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 81010, "status": "ok", "timestamp": 1602353988430, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="90af7df3-a745-4722-d77d-53f144212a91"
# evaluasi model memakai function
scaled = evaluate_model(models, X_train_selected_scaled_r, X_test_selected_scaled_r, y_train, y_test)
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
