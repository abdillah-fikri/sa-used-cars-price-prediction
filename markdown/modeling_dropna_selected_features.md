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
# Data Understanding
<!-- #endregion -->

```python id="B7_GzMp7H60k"
# !pip install -r requirements.txt
```

```python execution={"iopub.execute_input": "2020-10-08T14:24:16.108907Z", "iopub.status.busy": "2020-10-08T14:24:16.107910Z", "iopub.status.idle": "2020-10-08T14:24:21.655591Z", "shell.execute_reply": "2020-10-08T14:24:21.651602Z", "shell.execute_reply.started": "2020-10-08T14:24:16.108907Z"} id="vBMQYoVFZIqV"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils as explor
import plotly.graph_objects as go
import plotly.express as px
import missingno as msno
import category_encoders as ce
import miceforest as mf

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
```

```python execution={"iopub.execute_input": "2020-10-08T14:24:21.657585Z", "iopub.status.busy": "2020-10-08T14:24:21.657585Z", "iopub.status.idle": "2020-10-08T14:24:21.670553Z", "shell.execute_reply": "2020-10-08T14:24:21.666562Z", "shell.execute_reply.started": "2020-10-08T14:24:21.657585Z"} id="aIt9SQtmZIqZ"
sns.set(style='darkgrid', palette='muted')
```

```python cell_id="00002-1c7c9ecf-589f-4f15-9590-18a0d588c854" execution={"iopub.execute_input": "2020-10-08T14:24:21.677533Z", "iopub.status.busy": "2020-10-08T14:24:21.676535Z", "iopub.status.idle": "2020-10-08T14:24:21.746352Z", "shell.execute_reply": "2020-10-08T14:24:21.745353Z", "shell.execute_reply.started": "2020-10-08T14:24:21.677533Z"} id="S6ldr9t_Imtd" output_cleared=false
df = pd.read_csv('../data/raw/used_car_data.csv')
```

```python allow_embed=true cell_id="00003-bba23622-1282-44d4-b65f-d5ec13f504fa" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-08T14:24:21.749344Z", "iopub.status.busy": "2020-10-08T14:24:21.748346Z", "iopub.status.idle": "2020-10-08T14:24:21.811177Z", "shell.execute_reply": "2020-10-08T14:24:21.807187Z", "shell.execute_reply.started": "2020-10-08T14:24:21.749344Z"} executionInfo={"elapsed": 38921, "status": "ok", "timestamp": 1602465368029, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="oyIYOtRAIwWD" outputId="46c1af04-da35-4819-da56-e7ff7a5ca927" output_cleared=false
df.head()
```

```python cell_id="00004-051bff9d-c702-4146-a192-9dc23cb206dd" colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-08T14:24:21.815166Z", "iopub.status.busy": "2020-10-08T14:24:21.814169Z", "iopub.status.idle": "2020-10-08T14:24:21.842092Z", "shell.execute_reply": "2020-10-08T14:24:21.839103Z", "shell.execute_reply.started": "2020-10-08T14:24:21.815166Z"} executionInfo={"elapsed": 38905, "status": "ok", "timestamp": 1602465368030, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="t9SkWlvWIaiv" outputId="ead32c06-4246-4ef6-c361-9ef1d6cd9b08" output_cleared=false tags=[]
df.shape
```

```python cell_id="00005-214f7a33-5f2d-48cc-86a1-c41cd663c527" colab={"base_uri": "https://localhost:8080/", "height": 111} execution={"iopub.execute_input": "2020-10-08T14:24:21.845085Z", "iopub.status.busy": "2020-10-08T14:24:21.844088Z", "iopub.status.idle": "2020-10-08T14:24:21.905922Z", "shell.execute_reply": "2020-10-08T14:24:21.902933Z", "shell.execute_reply.started": "2020-10-08T14:24:21.845085Z"} executionInfo={"elapsed": 38872, "status": "ok", "timestamp": 1602465368031, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="8lVpvltpIai0" outputId="2b09a4e3-520c-4d31-a674-58d5dbc6704d" output_cleared=false tags=[]
df[df['Mileage'].isna()]
```

```python cell_id="00008-97aefa10-256d-4488-a2a5-34f8829ef33b" colab={"base_uri": "https://localhost:8080/", "height": 421} execution={"iopub.execute_input": "2020-10-08T14:24:21.908914Z", "iopub.status.busy": "2020-10-08T14:24:21.907917Z", "iopub.status.idle": "2020-10-08T14:24:21.968756Z", "shell.execute_reply": "2020-10-08T14:24:21.966758Z", "shell.execute_reply.started": "2020-10-08T14:24:21.908914Z"} executionInfo={"elapsed": 38845, "status": "ok", "timestamp": 1602465368032, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="BW8xd0boIajE" outputId="6ebbaaac-8ef1-4267-c850-e8ff26f78f18" output_cleared=false tags=[]
explor.null_checker(df)
```

```python cell_id="00009-41c0c52f-802c-4530-b39e-6b7dce61e76a" colab={"base_uri": "https://localhost:8080/", "height": 340} execution={"iopub.execute_input": "2020-10-08T14:24:21.971747Z", "iopub.status.busy": "2020-10-08T14:24:21.970748Z", "iopub.status.idle": "2020-10-08T14:24:22.000670Z", "shell.execute_reply": "2020-10-08T14:24:21.998674Z", "shell.execute_reply.started": "2020-10-08T14:24:21.970748Z"} executionInfo={"elapsed": 38827, "status": "ok", "timestamp": 1602465368033, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="2tMHFHl6MiKm" outputId="34a21015-7c30-4048-fc19-76a9e60c3af1" output_cleared=false tags=[]
df.info()
```

```python cell_id="00010-dd494b3f-977c-45a6-9a61-a44468a913b0" execution={"iopub.execute_input": "2020-10-08T14:24:22.010642Z", "iopub.status.busy": "2020-10-08T14:24:22.009645Z", "iopub.status.idle": "2020-10-08T14:24:22.063500Z", "shell.execute_reply": "2020-10-08T14:24:22.062503Z", "shell.execute_reply.started": "2020-10-08T14:24:22.010642Z"} id="s_ceUlxOKg7V" output_cleared=false
df['Brand'] = df['Name'].apply(lambda x: x.split(' ')[0])
df['Series'] = df['Name'].apply(lambda x: x.split(' ')[1])
df['Type'] = df['Name'].apply(lambda x: x.split(' ')[2])
df.drop(columns='Name', inplace=True)
```

```python cell_id="00011-a1625de7-7e59-420d-844b-5a3a2d1e652d" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-08T14:24:22.068490Z", "iopub.status.busy": "2020-10-08T14:24:22.067490Z", "iopub.status.idle": "2020-10-08T14:24:22.125335Z", "shell.execute_reply": "2020-10-08T14:24:22.124337Z", "shell.execute_reply.started": "2020-10-08T14:24:22.068490Z"} executionInfo={"elapsed": 38799, "status": "ok", "timestamp": 1602465368036, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="aY7bafplKu4K" outputId="88c26a14-3bcf-401e-c938-ed8e15072985" output_cleared=false
df.head()
```

```python cell_id="00012-d9063eee-15ac-4071-9fe3-e39d9dd7cf01" colab={"base_uri": "https://localhost:8080/", "height": 80} execution={"iopub.execute_input": "2020-10-08T14:24:22.128327Z", "iopub.status.busy": "2020-10-08T14:24:22.127330Z", "iopub.status.idle": "2020-10-08T14:24:22.156252Z", "shell.execute_reply": "2020-10-08T14:24:22.155254Z", "shell.execute_reply.started": "2020-10-08T14:24:22.128327Z"} executionInfo={"elapsed": 39402, "status": "ok", "timestamp": 1602465368665, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="VGanyS85IajY" outputId="1fe93683-058d-4e93-a586-c77aee46617e" output_cleared=false tags=[]
df[(df['Engine']=='72 CC')]
```

```python cell_id="00013-4aa8d13b-6dab-4f52-ae70-fc8f25367feb" colab={"base_uri": "https://localhost:8080/", "height": 68} execution={"iopub.execute_input": "2020-10-08T14:24:22.158246Z", "iopub.status.busy": "2020-10-08T14:24:22.158246Z", "iopub.status.idle": "2020-10-08T14:24:22.205122Z", "shell.execute_reply": "2020-10-08T14:24:22.202138Z", "shell.execute_reply.started": "2020-10-08T14:24:22.158246Z"} executionInfo={"elapsed": 39378, "status": "ok", "timestamp": 1602465368666, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="6m3Suen4Z_9Y" outputId="cb8c35be-1670-4514-dfe0-06208b33bf7a" output_cleared=false tags=[]
print('Satuan pada feature Mileage:', df['Mileage'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).unique())
print('Satuan pada feature Engine:', df['Engine'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).unique())
print('Satuan pada feature Power:', df['Power'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).unique())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 68} execution={"iopub.execute_input": "2020-10-08T14:24:22.207117Z", "iopub.status.busy": "2020-10-08T14:24:22.207117Z", "iopub.status.idle": "2020-10-08T14:24:22.251001Z", "shell.execute_reply": "2020-10-08T14:24:22.250002Z", "shell.execute_reply.started": "2020-10-08T14:24:22.207117Z"} executionInfo={"elapsed": 39354, "status": "ok", "timestamp": 1602465368667, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="IPkNA-lwZIrE" outputId="7c3ef55c-9a62-4353-f7a7-75ab785a6a63" tags=[]
print('Invalid Value pada feature Mileage:', pd.Series([x for x in df['Mileage'] if str(x).split(' ')[0].isalpha()]).unique())
print('Invalid Value pada feature Engine:', pd.Series([x for x in df['Engine'] if str(x).split(' ')[0].isalpha()]).unique())
print('Invalid Value pada feature Power:', pd.Series([x for x in df['Power'] if str(x).split(' ')[0].isalpha()]).unique())
```

```python cell_id="00014-3095d6e2-739a-4e9a-91dd-1b6a1aeecb0c" colab={"base_uri": "https://localhost:8080/", "height": 68} execution={"iopub.execute_input": "2020-10-08T14:24:22.253991Z", "iopub.status.busy": "2020-10-08T14:24:22.252994Z", "iopub.status.idle": "2020-10-08T14:24:22.282914Z", "shell.execute_reply": "2020-10-08T14:24:22.281917Z", "shell.execute_reply.started": "2020-10-08T14:24:22.253991Z"} executionInfo={"elapsed": 39337, "status": "ok", "timestamp": 1602465368669, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="fvUS80s_t_k8" outputId="c19af63e-f409-47c8-bdf5-3c0fd99985c1" output_cleared=false
df['Mileage'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).value_counts()
```

```python cell_id="00016-3c631710-d459-4dc7-854c-9b601a8f78a7" execution={"iopub.execute_input": "2020-10-08T14:24:22.285906Z", "iopub.status.busy": "2020-10-08T14:24:22.284910Z", "iopub.status.idle": "2020-10-08T14:24:22.374668Z", "shell.execute_reply": "2020-10-08T14:24:22.372673Z", "shell.execute_reply.started": "2020-10-08T14:24:22.285906Z"} id="3kGt8wrVLscx" output_cleared=false
df['Mileage (kmpl)'] = df['Mileage'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])
df['Engine (CC)'] = df['Engine'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])
df['Power (bhp)'] = df['Power'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])

df['Mileage (kmpl)'] = pd.to_numeric(df['Mileage (kmpl)'], errors='coerce')
df['Engine (CC)'] = pd.to_numeric(df['Engine (CC)'], errors='coerce')
df['Power (bhp)'] = pd.to_numeric(df['Power (bhp)'], errors='coerce')

df.drop(columns=['Mileage', 'Engine', 'Power'], inplace=True)
```

```python cell_id="00017-1cd20bd3-e6f2-422f-aceb-75c7b5c48651" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-08T14:24:22.377660Z", "iopub.status.busy": "2020-10-08T14:24:22.376663Z", "iopub.status.idle": "2020-10-08T14:24:22.418551Z", "shell.execute_reply": "2020-10-08T14:24:22.417553Z", "shell.execute_reply.started": "2020-10-08T14:24:22.377660Z"} executionInfo={"elapsed": 39311, "status": "ok", "timestamp": 1602465368671, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="L0OayFyZd12B" outputId="7a8d3c6a-8603-4a53-e57f-533528622e49" output_cleared=false
df.head()
```

```python allow_embed=false cell_id="00018-449a1211-5734-4bdb-a58d-5722fc129da5" colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-08T14:24:22.421542Z", "iopub.status.busy": "2020-10-08T14:24:22.420545Z", "iopub.status.idle": "2020-10-08T14:24:22.495347Z", "shell.execute_reply": "2020-10-08T14:24:22.494348Z", "shell.execute_reply.started": "2020-10-08T14:24:22.421542Z"} executionInfo={"elapsed": 39283, "status": "ok", "timestamp": 1602465368672, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="FMBbBCMFIajt" outputId="76104fa2-049c-456c-b124-0b36605d527c" output_cleared=false tags=[]
df.describe()
```

```python cell_id="00019-2ad88c07-d7d0-4fb4-a7b0-94c417269d0a" colab={"base_uri": "https://localhost:8080/", "height": 419} execution={"iopub.execute_input": "2020-10-08T14:24:22.498338Z", "iopub.status.busy": "2020-10-08T14:24:22.497340Z", "iopub.status.idle": "2020-10-08T14:24:22.575132Z", "shell.execute_reply": "2020-10-08T14:24:22.574136Z", "shell.execute_reply.started": "2020-10-08T14:24:22.498338Z"} executionInfo={"elapsed": 39254, "status": "ok", "timestamp": 1602465368673, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="PJFktTnEIajw" outputId="f3cb791d-28ec-4998-e832-94efcf712ad0" output_cleared=false tags=[]
df[df['Mileage (kmpl)']==0]
```

```python cell_id="00020-e43329af-54de-4465-87d6-cb3c4c39a087" colab={"base_uri": "https://localhost:8080/", "height": 80} execution={"iopub.execute_input": "2020-10-08T14:24:22.578125Z", "iopub.status.busy": "2020-10-08T14:24:22.577127Z", "iopub.status.idle": "2020-10-08T14:24:22.623004Z", "shell.execute_reply": "2020-10-08T14:24:22.622008Z", "shell.execute_reply.started": "2020-10-08T14:24:22.578125Z"} executionInfo={"elapsed": 39232, "status": "ok", "timestamp": 1602465368674, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="u6RCnI2OIajz" outputId="8278aefe-b8f7-421c-a5f6-e86443a54bd8" output_cleared=false tags=[]
df[df['Seats']==0]
```

```python cell_id="00021-e0395a10-0bb0-43fd-99ef-52126dbb0aa8" execution={"iopub.execute_input": "2020-10-08T14:24:22.625996Z", "iopub.status.busy": "2020-10-08T14:24:22.624999Z", "iopub.status.idle": "2020-10-08T14:24:22.654919Z", "shell.execute_reply": "2020-10-08T14:24:22.653921Z", "shell.execute_reply.started": "2020-10-08T14:24:22.625996Z"} id="Afk7mbdfIaj2" output_cleared=false tags=[]
df['Mileage (kmpl)'] = df['Mileage (kmpl)'].replace(0, np.nan)
df['Seats'] = df['Seats'].replace(0, np.nan)
```

```python cell_id="00022-bab6e9eb-e370-437d-bee9-837a51ebbb43" colab={"base_uri": "https://localhost:8080/", "height": 153} execution={"iopub.execute_input": "2020-10-08T14:24:22.657911Z", "iopub.status.busy": "2020-10-08T14:24:22.656913Z", "iopub.status.idle": "2020-10-08T14:24:22.702803Z", "shell.execute_reply": "2020-10-08T14:24:22.699800Z", "shell.execute_reply.started": "2020-10-08T14:24:22.657911Z"} executionInfo={"elapsed": 39201, "status": "ok", "timestamp": 1602465368677, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="wmm-4vqkIaj5" outputId="fe5583c0-17f9-406f-85cf-4bb6cd743485" output_cleared=false tags=[]
cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
df[cat_cols].nunique()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-08T14:24:22.705783Z", "iopub.status.busy": "2020-10-08T14:24:22.704786Z", "iopub.status.idle": "2020-10-08T14:24:22.717752Z", "shell.execute_reply": "2020-10-08T14:24:22.715757Z", "shell.execute_reply.started": "2020-10-08T14:24:22.705783Z"} executionInfo={"elapsed": 39177, "status": "ok", "timestamp": 1602465368678, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Q2UrO_LJZIri" outputId="446f883d-cf38-4ce8-9d01-dfe94a438bed"
['a', 'b']
```

```python cell_id="00023-826cb95b-08be-4095-8a0b-489a9f2a0fcc" colab={"base_uri": "https://localhost:8080/", "height": 1000} execution={"iopub.execute_input": "2020-10-08T14:24:22.721740Z", "iopub.status.busy": "2020-10-08T14:24:22.720743Z", "iopub.status.idle": "2020-10-08T14:24:22.748669Z", "shell.execute_reply": "2020-10-08T14:24:22.747671Z", "shell.execute_reply.started": "2020-10-08T14:24:22.721740Z"} executionInfo={"elapsed": 39156, "status": "ok", "timestamp": 1602465368679, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="YLT0EpfVlHZ2" outputId="ac81cd76-b286-4692-ac99-511ffec61664" output_cleared=false tags=[]
for col in cat_cols:
  print(col, df[col].unique(), '\n')
```

```python cell_id="00024-47c53a05-57f5-4f76-8ebc-626044400a24" execution={"iopub.execute_input": "2020-10-08T14:24:22.753658Z", "iopub.status.busy": "2020-10-08T14:24:22.751666Z", "iopub.status.idle": "2020-10-08T14:24:22.765623Z", "shell.execute_reply": "2020-10-08T14:24:22.763629Z", "shell.execute_reply.started": "2020-10-08T14:24:22.753658Z"} id="Ei1r1wmzlRHu" output_cleared=false
df['Brand'] = df['Brand'].replace('ISUZU', 'Isuzu')
```

```python cell_id="00025-57eb8332-9868-4559-adc8-cbaa48039fcd" colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-08T14:24:22.768623Z", "iopub.status.busy": "2020-10-08T14:24:22.767618Z", "iopub.status.idle": "2020-10-08T14:24:22.779586Z", "shell.execute_reply": "2020-10-08T14:24:22.778589Z", "shell.execute_reply.started": "2020-10-08T14:24:22.768623Z"} executionInfo={"elapsed": 39130, "status": "ok", "timestamp": 1602465368680, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="OXBlXSF3mZ4J" outputId="f5b8aa8a-57b6-4072-c3bc-b952d2ab1472" output_cleared=false tags=[]
print('Brand', df['Brand'].unique())
```

```python cell_id="00026-3a835b06-20e7-4253-852f-976a27b4f497" colab={"base_uri": "https://localhost:8080/", "height": 119} execution={"iopub.execute_input": "2020-10-08T14:24:22.782578Z", "iopub.status.busy": "2020-10-08T14:24:22.781582Z", "iopub.status.idle": "2020-10-08T14:24:22.796555Z", "shell.execute_reply": "2020-10-08T14:24:22.794548Z", "shell.execute_reply.started": "2020-10-08T14:24:22.782578Z"} executionInfo={"elapsed": 39110, "status": "ok", "timestamp": 1602465368681, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Q8rkmxpdm0Oz" outputId="16b396f0-0e28-411e-f63e-f74b4d206a7e" output_cleared=false
df['Fuel_Type'].value_counts()
```

```python cell_id="00027-bbfa9e1f-1505-4f93-8bb1-e55ff6cae887" colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-08T14:24:22.816487Z", "iopub.status.busy": "2020-10-08T14:24:22.815490Z", "iopub.status.idle": "2020-10-08T14:24:22.843415Z", "shell.execute_reply": "2020-10-08T14:24:22.841420Z", "shell.execute_reply.started": "2020-10-08T14:24:22.816487Z"} executionInfo={"elapsed": 39085, "status": "ok", "timestamp": 1602465368681, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="zQVZX-GcIakM" outputId="9321da87-2acc-4686-8f68-570290a38ba9" output_cleared=false tags=[]
df['Series'].nunique()
```

```python cell_id="00030-25fa031b-975c-43ae-9e06-a422b44e4789" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-08T14:24:22.893160Z", "iopub.status.busy": "2020-10-08T14:24:22.892160Z", "iopub.status.idle": "2020-10-08T14:24:22.937041Z", "shell.execute_reply": "2020-10-08T14:24:22.935050Z", "shell.execute_reply.started": "2020-10-08T14:24:22.893160Z"} executionInfo={"elapsed": 39037, "status": "ok", "timestamp": 1602465368682, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="89WdS5GnIakU" outputId="9895dbab-9907-407a-bd8f-d916537be5c9" output_cleared=false tags=[]
df.head()
```

```python cell_id="00031-7ddc6533-c13e-4779-b1f0-09774747357f" colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-08T14:24:22.940032Z", "iopub.status.busy": "2020-10-08T14:24:22.940032Z", "iopub.status.idle": "2020-10-08T14:24:23.016826Z", "shell.execute_reply": "2020-10-08T14:24:23.014834Z", "shell.execute_reply.started": "2020-10-08T14:24:22.940032Z"} executionInfo={"elapsed": 39366, "status": "ok", "timestamp": 1602465369037, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Ney4gM1WIakY" outputId="e953f5ce-90a9-439f-8090-4344dbef82f0" output_cleared=false tags=[]
df.describe()
```

```python cell_id="00032-a59a0124-6b41-4b36-aead-c0d4df135d9b" colab={"base_uri": "https://localhost:8080/", "height": 173} execution={"iopub.execute_input": "2020-10-08T14:24:23.018820Z", "iopub.status.busy": "2020-10-08T14:24:23.017823Z", "iopub.status.idle": "2020-10-08T14:24:23.079172Z", "shell.execute_reply": "2020-10-08T14:24:23.078194Z", "shell.execute_reply.started": "2020-10-08T14:24:23.018820Z"} executionInfo={"elapsed": 39343, "status": "ok", "timestamp": 1602465369039, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="3Jk-RIU6Iakb" outputId="b9947bfe-63b6-4962-de20-bb3306266eb6" output_cleared=false tags=[]
df.describe(include=['object']) 
```

```python colab={"base_uri": "https://localhost:8080/", "height": 483} execution={"iopub.execute_input": "2020-10-08T14:24:23.081170Z", "iopub.status.busy": "2020-10-08T14:24:23.080170Z", "iopub.status.idle": "2020-10-08T14:24:23.110089Z", "shell.execute_reply": "2020-10-08T14:24:23.109091Z", "shell.execute_reply.started": "2020-10-08T14:24:23.081170Z"} executionInfo={"elapsed": 39316, "status": "ok", "timestamp": 1602465369040, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="PMAtruMZZIsX" outputId="e8bb2232-a85e-4bbe-943c-d0840fa382da"
explor.null_checker(df)
```

```python cell_id="00034-a74bd3e4-e566-4425-a0a5-cc1c9af4f640" execution={"iopub.execute_input": "2020-10-08T14:24:23.112084Z", "iopub.status.busy": "2020-10-08T14:24:23.112084Z", "iopub.status.idle": "2020-10-08T14:24:23.125050Z", "shell.execute_reply": "2020-10-08T14:24:23.124051Z", "shell.execute_reply.started": "2020-10-08T14:24:23.112084Z"} id="LbW9gLQVIakg" output_cleared=false tags=[]
df.loc[df['Fuel_Type']=='Electric', 'Mileage (kmpl)'] = df.loc[df['Fuel_Type']=='Electric', 'Mileage (kmpl)'].replace(np.nan, 0)
```

```python cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" colab={"base_uri": "https://localhost:8080/", "height": 111} execution={"iopub.execute_input": "2020-10-08T14:24:23.127043Z", "iopub.status.busy": "2020-10-08T14:24:23.126046Z", "iopub.status.idle": "2020-10-08T14:24:23.155969Z", "shell.execute_reply": "2020-10-08T14:24:23.153972Z", "shell.execute_reply.started": "2020-10-08T14:24:23.127043Z"} executionInfo={"elapsed": 39278, "status": "ok", "timestamp": 1602465369042, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="bKob_zWgIakl" outputId="a60a5341-5df5-4af0-c4c1-e21a44d90fc2" output_cleared=false tags=[]
df.loc[df['Fuel_Type']=='Electric']
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
explor.null_checker(df)
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
encodes = ['Location','Fuel_Type','Transmission','Owner_Type', 'Brand']
encoder = ce.OneHotEncoder(cols=encodes,
                          use_cat_names=True)
encoder.fit(X_train)

# encoding train set
X_train = encoder.transform(X_train)

# encoding test set
X_test = encoder.transform(X_test)
```

```python execution={"iopub.execute_input": "2020-10-08T14:24:52.214085Z", "iopub.status.busy": "2020-10-08T14:24:52.213088Z", "iopub.status.idle": "2020-10-08T14:24:52.385628Z", "shell.execute_reply": "2020-10-08T14:24:52.384657Z", "shell.execute_reply.started": "2020-10-08T14:24:52.213088Z"} id="kcMLnvJxZIuD"
# Target encoding/One hot encoding untuk feature dengan kategori yang banyak
encodes = ['Series','Type']
target_encodes = ce.TargetEncoder(cols= encodes)
target_encodes.fit(X_train,y_train)

# Encoding train set
X_train = target_encodes.transform(X_train)

# Encoding test set
X_test = target_encodes.transform(X_test)
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

Dengan adanya pencilan, StandardScaler tidak menjamin skala fitur yang seimbang, karena pengaruh pencilan saat menghitung rata-rata empiris dan deviasi standar. Hal ini menyebabkan penyusutan kisaran nilai fitur.

```python id="2lQZQbORMwYB"
# Scaling data with standard scaller
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_selected)
X_train_selected_scaled = scaler.transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 81010, "status": "ok", "timestamp": 1602353988430, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="90af7df3-a745-4722-d77d-53f144212a91"
# evaluasi model memakai function
evaluate_model(models, X_train_selected_scaled, X_test_selected_scaled, y_train, y_test)
```

```python id="2lQZQbORMwYB"
# Scaling data with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train_selected)
X_train_selected_scaled_m = scaler.transform(X_train_selected)
X_test_selected_scaled_m = scaler.transform(X_test_selected)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 297} executionInfo={"elapsed": 81010, "status": "ok", "timestamp": 1602353988430, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="90af7df3-a745-4722-d77d-53f144212a91"
# evaluasi model memakai function
evaluate_model(models, X_train_selected_scaled_m, X_test_selected_scaled_m, y_train, y_test)
```

RobustScaler mengurangi median kolom dan membaginya dengan rentang interkuartil.

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

dropna_selected = pd.concat([unscaled, scaled], axis=0)
dropna_selected
```

```python
dropna_selected.to_csv('dropna_selected.csv')
```
