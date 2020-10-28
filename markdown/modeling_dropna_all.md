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
        hash: 8d4d772f21767a3a72f3356b4ab1badff3b831eb21eba306d4ebdf1fe7777d12
    name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
---

<!-- #region id="IN1jfOnfZIqT" -->
# Data Importing
<!-- #endregion -->

```python cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" colab={"base_uri": "https://localhost:8080/", "height": 111} execution={"iopub.execute_input": "2020-10-13T13:29:52.833962Z", "iopub.status.busy": "2020-10-13T13:29:52.833962Z", "iopub.status.idle": "2020-10-13T13:29:52.864878Z", "shell.execute_reply": "2020-10-13T13:29:52.863881Z", "shell.execute_reply.started": "2020-10-13T13:29:52.833962Z"} executionInfo={"elapsed": 5713, "status": "ok", "timestamp": 1602555650537, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="bKob_zWgIakl" outputId="6062b1e9-6be6-48c6-f955-0d621e64a663" output_cleared=false tags=[]
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

```python colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-13T13:29:52.867870Z", "iopub.status.busy": "2020-10-13T13:29:52.866875Z", "iopub.status.idle": "2020-10-13T13:29:52.879124Z", "shell.execute_reply": "2020-10-13T13:29:52.879124Z", "shell.execute_reply.started": "2020-10-13T13:29:52.867870Z"} executionInfo={"elapsed": 5696, "status": "ok", "timestamp": 1602555650538, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="INV8VvOYZItN" outputId="6cf7dcec-3532-4b56-dd17-44ca7978010d"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 483} execution={"iopub.execute_input": "2020-10-13T13:29:52.879124Z", "iopub.status.busy": "2020-10-13T13:29:52.879124Z", "iopub.status.idle": "2020-10-13T13:29:52.910917Z", "shell.execute_reply": "2020-10-13T13:29:52.909951Z", "shell.execute_reply.started": "2020-10-13T13:29:52.879124Z"} executionInfo={"elapsed": 5665, "status": "ok", "timestamp": 1602555650538, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="TYqvFHW1HqFX" outputId="6ccbecc5-7a77-4f56-9ed9-d63da75e6742"
# Drop missing values
df = df.dropna()
null_checker(df)
```

<!-- #region id="yEgVyyNSZIt9" -->
## Train test split
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-13T13:29:52.911913Z", "iopub.status.busy": "2020-10-13T13:29:52.911913Z", "iopub.status.idle": "2020-10-13T13:29:52.926873Z", "shell.execute_reply": "2020-10-13T13:29:52.925908Z", "shell.execute_reply.started": "2020-10-13T13:29:52.911913Z"} executionInfo={"elapsed": 875, "status": "ok", "timestamp": 1602555655449, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="nPxFt6bSZIt-"
# melakukan train test split di awal untuk mencegah data bocor ke test set saat dilakukan encoding/imputation
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-13T13:29:52.928873Z", "iopub.status.busy": "2020-10-13T13:29:52.927872Z", "iopub.status.idle": "2020-10-13T13:29:53.107446Z", "shell.execute_reply": "2020-10-13T13:29:53.106483Z", "shell.execute_reply.started": "2020-10-13T13:29:52.928873Z"} executionInfo={"elapsed": 776, "status": "ok", "timestamp": 1602555727773, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="_0criLnZIakn" outputId="8b1555e3-4ca7-4bc9-c310-79d7840c1aa1" output_cleared=false tags=[]
# One hot encoding
col_to_encode = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
oh_encoder = ce.OneHotEncoder(cols=col_to_encode,
                              use_cat_names=True)
oh_encoder.fit(X_train)

# Encoding train set
X_train = oh_encoder.transform(X_train)
# Encoding test set
X_test = oh_encoder.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-13T13:29:53.108444Z", "iopub.status.busy": "2020-10-13T13:29:53.108444Z", "iopub.status.idle": "2020-10-13T13:29:53.178943Z", "shell.execute_reply": "2020-10-13T13:29:53.178943Z", "shell.execute_reply.started": "2020-10-13T13:29:53.108444Z"} executionInfo={"elapsed": 856, "status": "ok", "timestamp": 1602555730207, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="kcMLnvJxZIuD" outputId="0f9c7677-a896-4027-9610-562e404a18b4"
# Target encoding/One hot encoding untuk feature dengan kategori yang banyak
col_to_encode = ['Series', 'Type']
target_encoder = ce.TargetEncoder(cols=col_to_encode)
target_encoder.fit(X_train, y_train)

# Encoding train set
X_train = target_encoder.transform(X_train)
# Encoding test set
X_test = target_encoder.transform(X_test)
```

<!-- #region id="wV2sjkqEZIup" -->
# Modeling
<!-- #endregion -->

<!-- #region id="4g_nWqotKl6_" -->
## Functions
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-13T13:29:53.181043Z", "iopub.status.busy": "2020-10-13T13:29:53.181043Z", "iopub.status.idle": "2020-10-13T13:29:53.195221Z", "shell.execute_reply": "2020-10-13T13:29:53.194224Z", "shell.execute_reply.started": "2020-10-13T13:29:53.181043Z"} executionInfo={"elapsed": 984, "status": "ok", "timestamp": 1602555740977, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Qp4QHIuFZIuq"
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

```python execution={"iopub.execute_input": "2020-10-13T13:29:53.198250Z", "iopub.status.busy": "2020-10-13T13:29:53.197220Z", "iopub.status.idle": "2020-10-13T13:29:53.212177Z", "shell.execute_reply": "2020-10-13T13:29:53.210182Z", "shell.execute_reply.started": "2020-10-13T13:29:53.198250Z"} executionInfo={"elapsed": 837, "status": "ok", "timestamp": 1602556656550, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="BXEr8F5VZIu0"
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

```python execution={"iopub.execute_input": "2020-10-13T13:34:36.180450Z", "iopub.status.busy": "2020-10-13T13:34:36.179421Z", "iopub.status.idle": "2020-10-13T13:34:36.198368Z", "shell.execute_reply": "2020-10-13T13:34:36.197370Z", "shell.execute_reply.started": "2020-10-13T13:34:36.180450Z"} executionInfo={"elapsed": 802, "status": "ok", "timestamp": 1602556659028, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Oux2OxeDZIu2"
tree_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor()
xgb_model = XGBRegressor(objective='reg:squarederror')
lgb_model = LGBMRegressor()
cat_model = CatBoostRegressor(verbose=0, iterations=2000)
lr_model = LinearRegression()
lasso_model = Lasso()

models_tree = {'DecisionTreeRegressor' : tree_model,
          'RandomForestRegressor' : rf_model,
          'XGBRegressor' : xgb_model,
          'CatBoostRegressor' : cat_model,
          'LGBMRegressor' : lgb_model}

models_linear ={'LinearRegression': lr_model,
          'LassoRegression': lasso_model}
```

<!-- #region id="kCSEOF35MoSB" -->
### Unscaled dataset
<!-- #endregion -->

```python
evaluate_model(models_tree, X_train, X_test, y_train, y_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-13T13:48:49.354143Z", "iopub.status.busy": "2020-10-13T13:48:49.354143Z", "iopub.status.idle": "2020-10-13T13:49:38.126193Z", "shell.execute_reply": "2020-10-13T13:49:38.125196Z", "shell.execute_reply.started": "2020-10-13T13:48:49.354143Z"} executionInfo={"elapsed": 43072, "status": "ok", "timestamp": 1602556705466, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="53cf5ba8-9d0d-44eb-c0f5-0e1c8d77f42f"
# evaluasi model memakai function
unscaled = evaluate_model(models_tree, X_train, X_test, y_train, y_test)
```

<!-- #region id="AodaQJBNMtob" -->
### Scaled dataset
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-13T13:49:38.129185Z", "iopub.status.busy": "2020-10-13T13:49:38.129185Z", "iopub.status.idle": "2020-10-13T13:49:38.201992Z", "shell.execute_reply": "2020-10-13T13:49:38.200992Z", "shell.execute_reply.started": "2020-10-13T13:49:38.129185Z"} id="HUYPPvWk2oo_"
# Scaling data
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python execution={"iopub.execute_input": "2020-10-13T13:49:38.203987Z", "iopub.status.busy": "2020-10-13T13:49:38.202989Z", "iopub.status.idle": "2020-10-13T13:50:28.224677Z", "shell.execute_reply": "2020-10-13T13:50:28.223681Z", "shell.execute_reply.started": "2020-10-13T13:49:38.203987Z"}
# evaluasi model memakai function
scaled = evaluate_model(models_tree, X_train_scaled, X_test_scaled, y_train, y_test)
```

### Summarizing

```python
unscaled['Dataset Version'] = 'dropna + all + unscaled'
scaled['Dataset Version'] = 'dropna + all + scaled'
```

```python
dropna_all = pd.concat([unscaled, scaled], axis=0)
dropna_all
```

```python
dropna_all.to_csv('../data/processed/summary_dropna_all.csv')
```
