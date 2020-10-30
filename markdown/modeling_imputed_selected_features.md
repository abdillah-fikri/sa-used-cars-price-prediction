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

```python
import numpy as np
import pandas as pd
import category_encoders as ce
import miceforest as mf

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

```python colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-15T12:54:13.988187Z", "iopub.status.busy": "2020-10-15T12:54:13.987190Z", "iopub.status.idle": "2020-10-15T12:54:14.002152Z", "shell.execute_reply": "2020-10-15T12:54:14.001155Z", "shell.execute_reply.started": "2020-10-15T12:54:13.988187Z"} executionInfo={"elapsed": 8249, "status": "ok", "timestamp": 1602557855909, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="INV8VvOYZItN" outputId="d4ca600b-dc4f-4db2-fba8-54aa6678f0c0"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
```

<!-- #region id="yEgVyyNSZIt9" -->
## Train test split
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:54:14.005145Z", "iopub.status.busy": "2020-10-15T12:54:14.004145Z", "iopub.status.idle": "2020-10-15T12:54:14.033068Z", "shell.execute_reply": "2020-10-15T12:54:14.032071Z", "shell.execute_reply.started": "2020-10-15T12:54:14.005145Z"} executionInfo={"elapsed": 8232, "status": "ok", "timestamp": 1602557855911, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="nPxFt6bSZIt-"
# melakukan train test split di awal untuk mencegah data leakage
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-15T12:54:14.034066Z", "iopub.status.busy": "2020-10-15T12:54:14.034066Z", "iopub.status.idle": "2020-10-15T12:54:14.187654Z", "shell.execute_reply": "2020-10-15T12:54:14.186657Z", "shell.execute_reply.started": "2020-10-15T12:54:14.034066Z"} executionInfo={"elapsed": 1054, "status": "ok", "timestamp": 1602557861674, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="_0criLnZIakn" outputId="766c5c78-5fac-492b-e39c-674c73139932" output_cleared=false tags=[]
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

```python colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-15T12:54:14.188652Z", "iopub.status.busy": "2020-10-15T12:54:14.188652Z", "iopub.status.idle": "2020-10-15T12:54:14.267439Z", "shell.execute_reply": "2020-10-15T12:54:14.266443Z", "shell.execute_reply.started": "2020-10-15T12:54:14.188652Z"} executionInfo={"elapsed": 587, "status": "ok", "timestamp": 1602557861677, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="kcMLnvJxZIuD" outputId="ec506ea3-a38a-4b80-9e62-2f3af531162a"
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

<!-- #region id="6MJs1hK7Iv1N" -->
## Missing Value Imputation
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:54:14.269434Z", "iopub.status.busy": "2020-10-15T12:54:14.269434Z", "iopub.status.idle": "2020-10-15T12:54:25.487160Z", "shell.execute_reply": "2020-10-15T12:54:25.487160Z", "shell.execute_reply.started": "2020-10-15T12:54:14.269434Z"} executionInfo={"elapsed": 9747, "status": "ok", "timestamp": 1602558096090, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ccgkETh_Iv1O"
# memprediksi nilai missing value dengan algoritma 
imputer = mf.KernelDataSet(
  X_train,
  save_all_iterations=True,
  random_state=1991,
  mean_match_candidates=5
)
imputer.mice(10)
```

```python execution={"iopub.execute_input": "2020-10-15T12:54:25.487160Z", "iopub.status.busy": "2020-10-15T12:54:25.487160Z", "iopub.status.idle": "2020-10-15T12:54:25.503631Z", "shell.execute_reply": "2020-10-15T12:54:25.502669Z", "shell.execute_reply.started": "2020-10-15T12:54:25.487160Z"} executionInfo={"elapsed": 769, "status": "ok", "timestamp": 1602558116061, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="e_zrbZk6Iv1S"
# Train set imputation
X_train_full = imputer.complete_data()
```

```python execution={"iopub.execute_input": "2020-10-15T12:54:25.505624Z", "iopub.status.busy": "2020-10-15T12:54:25.504627Z", "iopub.status.idle": "2020-10-15T12:54:27.936064Z", "shell.execute_reply": "2020-10-15T12:54:27.936064Z", "shell.execute_reply.started": "2020-10-15T12:54:25.505624Z"} executionInfo={"elapsed": 2626, "status": "ok", "timestamp": 1602558147720, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="s3TrxVjQIv1Z"
# Test set imputation
new_data = imputer.impute_new_data(X_test)
X_test_full = new_data.complete_data()
```

<!-- #region id="zYZVKzQYIxVx" -->
## Feature Selection
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:54:27.936064Z", "iopub.status.busy": "2020-10-15T12:54:27.936064Z", "iopub.status.idle": "2020-10-15T12:54:28.031986Z", "shell.execute_reply": "2020-10-15T12:54:28.030987Z", "shell.execute_reply.started": "2020-10-15T12:54:27.936064Z"} executionInfo={"elapsed": 974, "status": "ok", "timestamp": 1602558988123, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="5RY-BaL8IxVy"
# Memfilter feature dengan korelasi tinggi
corr_price = X_train.join(y_train).corr()['Price']
index = corr_price[(corr_price < -0.20) | (corr_price > 0.20)].index

X_train_full =  X_train_full[index[:-1]]
X_test_full = X_test_full[index[:-1]]
```

<!-- #region id="wV2sjkqEZIup" -->
# Modeling
<!-- #endregion -->

<!-- #region id="aR4Sp3UCZIu2" -->
## Base Model
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:54:28.065895Z", "iopub.status.busy": "2020-10-15T12:54:28.065895Z", "iopub.status.idle": "2020-10-15T12:54:28.079856Z", "shell.execute_reply": "2020-10-15T12:54:28.077864Z", "shell.execute_reply.started": "2020-10-15T12:54:28.065895Z"} executionInfo={"elapsed": 678, "status": "ok", "timestamp": 1602559134050, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Oux2OxeDZIu2"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:54:28.081881Z", "iopub.status.busy": "2020-10-15T12:54:28.080854Z", "iopub.status.idle": "2020-10-15T12:55:11.188310Z", "shell.execute_reply": "2020-10-15T12:55:11.187312Z", "shell.execute_reply.started": "2020-10-15T12:54:28.081881Z"} executionInfo={"elapsed": 30383, "status": "ok", "timestamp": 1602559165523, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="857f512d-6910-4625-e01b-c6b587a9094c"
# evaluasi model memakai function
unscaled = evaluate_model(models, X_train_full, X_test_full, y_train, y_test)
```

<!-- #region id="AodaQJBNMtob" -->
### Scaled dataset
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:55:11.191302Z", "iopub.status.busy": "2020-10-15T12:55:11.190305Z", "iopub.status.idle": "2020-10-15T12:55:11.236183Z", "shell.execute_reply": "2020-10-15T12:55:11.235184Z", "shell.execute_reply.started": "2020-10-15T12:55:11.191302Z"} executionInfo={"elapsed": 25276, "status": "ok", "timestamp": 1602559165525, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="2lQZQbORMwYB"
# Scaling data
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train_full)
X_train_full_scaled = scaler.transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:55:11.239174Z", "iopub.status.busy": "2020-10-15T12:55:11.238177Z", "iopub.status.idle": "2020-10-15T12:55:54.767071Z", "shell.execute_reply": "2020-10-15T12:55:54.767071Z", "shell.execute_reply.started": "2020-10-15T12:55:11.239174Z"} executionInfo={"elapsed": 54513, "status": "ok", "timestamp": 1602559195270, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="06962bd1-1bb2-4c3e-bd1c-74e71a1d0ed5"
# evaluasi model memakai function
scaled = evaluate_model(models, X_train_full_scaled, X_test_full_scaled, y_train, y_test)
```

<!-- #region id="bg_vcQxLLg0n" -->
### Summarizing
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:55:54.771050Z", "iopub.status.busy": "2020-10-15T12:55:54.770053Z", "iopub.status.idle": "2020-10-15T12:55:54.784016Z", "shell.execute_reply": "2020-10-15T12:55:54.783018Z", "shell.execute_reply.started": "2020-10-15T12:55:54.771050Z"}
unscaled['Dataset Version'] = 'imputed + selected + unscaled'
scaled['Dataset Version'] = 'imputed + selected + scaled'
```

```python execution={"iopub.execute_input": "2020-10-15T12:55:54.786011Z", "iopub.status.busy": "2020-10-15T12:55:54.786011Z", "iopub.status.idle": "2020-10-15T12:55:54.831887Z", "shell.execute_reply": "2020-10-15T12:55:54.830889Z", "shell.execute_reply.started": "2020-10-15T12:55:54.786011Z"}
imputed_selected = pd.concat([unscaled, scaled], axis=0)
imputed_selected
```

```python execution={"iopub.execute_input": "2020-10-15T12:55:54.834878Z", "iopub.status.busy": "2020-10-15T12:55:54.833882Z", "iopub.status.idle": "2020-10-15T12:55:54.847844Z", "shell.execute_reply": "2020-10-15T12:55:54.846847Z", "shell.execute_reply.started": "2020-10-15T12:55:54.834878Z"}
imputed_selected.to_csv('../data/processed/summary_imputed_selected.csv')
```
