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

```python cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" colab={"base_uri": "https://localhost:8080/", "height": 111} execution={"iopub.execute_input": "2020-10-13T13:29:52.833962Z", "iopub.status.busy": "2020-10-13T13:29:52.833962Z", "iopub.status.idle": "2020-10-13T13:29:52.864878Z", "shell.execute_reply": "2020-10-13T13:29:52.863881Z", "shell.execute_reply.started": "2020-10-13T13:29:52.833962Z"} executionInfo={"elapsed": 5713, "status": "ok", "timestamp": 1602555650537, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="bKob_zWgIakl" outputId="6062b1e9-6be6-48c6-f955-0d621e64a663" output_cleared=false tags=[]
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
# melakukan train test split di awal untuk mencegah data leakage
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-13T13:29:52.928873Z", "iopub.status.busy": "2020-10-13T13:29:52.927872Z", "iopub.status.idle": "2020-10-13T13:29:53.107446Z", "shell.execute_reply": "2020-10-13T13:29:53.106483Z", "shell.execute_reply.started": "2020-10-13T13:29:52.928873Z"} executionInfo={"elapsed": 776, "status": "ok", "timestamp": 1602555727773, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="_0criLnZIakn" outputId="8b1555e3-4ca7-4bc9-c310-79d7840c1aa1" output_cleared=false tags=[]
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

```python
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

```python colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-13T13:29:53.108444Z", "iopub.status.busy": "2020-10-13T13:29:53.108444Z", "iopub.status.idle": "2020-10-13T13:29:53.178943Z", "shell.execute_reply": "2020-10-13T13:29:53.178943Z", "shell.execute_reply.started": "2020-10-13T13:29:53.108444Z"} executionInfo={"elapsed": 856, "status": "ok", "timestamp": 1602555730207, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="kcMLnvJxZIuD" outputId="0f9c7677-a896-4027-9610-562e404a18b4"
# Target encoding for high cardinality feature
col_to_encode = X_train.select_dtypes("object").columns
encoder = ce.TargetEncoder(cols=col_to_encode)
encoder.fit(X_train, y_train)

# Encoding train set
X_train = encoder.transform(X_train)
# Encoding test set
X_test = encoder.transform(X_test)
```

<!-- #region id="wV2sjkqEZIup" -->
# Modeling
<!-- #endregion -->

<!-- #region id="aR4Sp3UCZIu2" -->
## Base Model
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-13T13:34:36.180450Z", "iopub.status.busy": "2020-10-13T13:34:36.179421Z", "iopub.status.idle": "2020-10-13T13:34:36.198368Z", "shell.execute_reply": "2020-10-13T13:34:36.197370Z", "shell.execute_reply.started": "2020-10-13T13:34:36.180450Z"} executionInfo={"elapsed": 802, "status": "ok", "timestamp": 1602556659028, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Oux2OxeDZIu2"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-13T13:48:49.354143Z", "iopub.status.busy": "2020-10-13T13:48:49.354143Z", "iopub.status.idle": "2020-10-13T13:49:38.126193Z", "shell.execute_reply": "2020-10-13T13:49:38.125196Z", "shell.execute_reply.started": "2020-10-13T13:48:49.354143Z"} executionInfo={"elapsed": 43072, "status": "ok", "timestamp": 1602556705466, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="53cf5ba8-9d0d-44eb-c0f5-0e1c8d77f42f"
# evaluasi model memakai function
unscaled = evaluate_model(models, X_train, X_test, y_train, y_test)
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
scaled = evaluate_model(models, X_train_scaled, X_test_scaled, y_train, y_test)
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
