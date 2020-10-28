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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="IN1jfOnfZIqT"
# # Data Understanding

# %% colab={"base_uri": "https://localhost:8080/", "height": 870} executionInfo={"elapsed": 4659, "status": "ok", "timestamp": 1602557851826, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="B7_GzMp7H60k" outputId="d637067a-0f84-4b12-d32d-b0346a9f008c"
# # !pip install -r requirements.txt

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.status.idle": "2020-10-15T12:54:12.762613Z", "shell.execute_reply": "2020-10-15T12:54:12.762613Z", "shell.execute_reply.started": "2020-10-15T12:54:08.538935Z"} executionInfo={"elapsed": 7454, "status": "ok", "timestamp": 1602557854624, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="vBMQYoVFZIqV" outputId="06bf21de-424e-4339-e9db-7861d53fd06a"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_exploration as explor
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

# %% execution={"iopub.execute_input": "2020-10-15T12:54:12.762613Z", "iopub.status.busy": "2020-10-15T12:54:12.762613Z", "iopub.status.idle": "2020-10-15T12:54:12.796610Z", "shell.execute_reply": "2020-10-15T12:54:12.793619Z", "shell.execute_reply.started": "2020-10-15T12:54:12.762613Z"} executionInfo={"elapsed": 7451, "status": "ok", "timestamp": 1602557854626, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="aIt9SQtmZIqZ"
sns.set(style='darkgrid', palette='muted')

# %% cell_id="00002-1c7c9ecf-589f-4f15-9590-18a0d588c854" execution={"iopub.execute_input": "2020-10-15T12:54:12.811571Z", "iopub.status.busy": "2020-10-15T12:54:12.810575Z", "iopub.status.idle": "2020-10-15T12:54:12.890353Z", "shell.execute_reply": "2020-10-15T12:54:12.889391Z", "shell.execute_reply.started": "2020-10-15T12:54:12.811571Z"} executionInfo={"elapsed": 7448, "status": "ok", "timestamp": 1602557854628, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="S6ldr9t_Imtd" output_cleared=false
df = pd.read_csv('used_car_data.csv')

# %% allow_embed=true cell_id="00003-bba23622-1282-44d4-b65f-d5ec13f504fa" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-15T12:54:12.892347Z", "iopub.status.busy": "2020-10-15T12:54:12.891350Z", "iopub.status.idle": "2020-10-15T12:54:12.938224Z", "shell.execute_reply": "2020-10-15T12:54:12.937227Z", "shell.execute_reply.started": "2020-10-15T12:54:12.892347Z"} executionInfo={"elapsed": 7433, "status": "ok", "timestamp": 1602557854630, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="oyIYOtRAIwWD" outputId="81d6ceaf-4424-4f11-9a9f-05f2a2bd9a01" output_cleared=false
df.head()

# %% cell_id="00004-051bff9d-c702-4146-a192-9dc23cb206dd" colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-15T12:54:12.940220Z", "iopub.status.busy": "2020-10-15T12:54:12.940220Z", "iopub.status.idle": "2020-10-15T12:54:12.955180Z", "shell.execute_reply": "2020-10-15T12:54:12.952193Z", "shell.execute_reply.started": "2020-10-15T12:54:12.940220Z"} executionInfo={"elapsed": 7418, "status": "ok", "timestamp": 1602557854632, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="t9SkWlvWIaiv" outputId="36bd118f-e883-4bfa-af80-1cb61b74b3f5" output_cleared=false tags=[]
df.shape

# %% cell_id="00005-214f7a33-5f2d-48cc-86a1-c41cd663c527" colab={"base_uri": "https://localhost:8080/", "height": 111} execution={"iopub.execute_input": "2020-10-15T12:54:12.957174Z", "iopub.status.busy": "2020-10-15T12:54:12.956177Z", "iopub.status.idle": "2020-10-15T12:54:13.046934Z", "shell.execute_reply": "2020-10-15T12:54:13.045935Z", "shell.execute_reply.started": "2020-10-15T12:54:12.957174Z"} executionInfo={"elapsed": 7392, "status": "ok", "timestamp": 1602557854634, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="8lVpvltpIai0" outputId="89f2afa2-d987-46e5-fbfc-308b8ae2be3e" output_cleared=false tags=[]
df[df['Mileage'].isna()]

# %% cell_id="00008-97aefa10-256d-4488-a2a5-34f8829ef33b" colab={"base_uri": "https://localhost:8080/", "height": 421} execution={"iopub.execute_input": "2020-10-15T12:54:13.049925Z", "iopub.status.busy": "2020-10-15T12:54:13.049925Z", "iopub.status.idle": "2020-10-15T12:54:13.093810Z", "shell.execute_reply": "2020-10-15T12:54:13.091816Z", "shell.execute_reply.started": "2020-10-15T12:54:13.049925Z"} executionInfo={"elapsed": 7716, "status": "ok", "timestamp": 1602557854979, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="BW8xd0boIajE" outputId="2cb72400-0ff0-4d74-ca2a-7b3deb81908c" output_cleared=false tags=[]
explor.null_checker(df)

# %% cell_id="00009-41c0c52f-802c-4530-b39e-6b7dce61e76a" colab={"base_uri": "https://localhost:8080/", "height": 340} execution={"iopub.execute_input": "2020-10-15T12:54:13.098804Z", "iopub.status.busy": "2020-10-15T12:54:13.096801Z", "iopub.status.idle": "2020-10-15T12:54:13.123728Z", "shell.execute_reply": "2020-10-15T12:54:13.122731Z", "shell.execute_reply.started": "2020-10-15T12:54:13.098804Z"} executionInfo={"elapsed": 7700, "status": "ok", "timestamp": 1602557854981, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="2tMHFHl6MiKm" outputId="f538eb58-48f5-446c-cdad-d092dfa2a4b9" output_cleared=false tags=[]
df.info()

# %% cell_id="00010-dd494b3f-977c-45a6-9a61-a44468a913b0" execution={"iopub.execute_input": "2020-10-15T12:54:13.126720Z", "iopub.status.busy": "2020-10-15T12:54:13.125723Z", "iopub.status.idle": "2020-10-15T12:54:13.169605Z", "shell.execute_reply": "2020-10-15T12:54:13.168607Z", "shell.execute_reply.started": "2020-10-15T12:54:13.126720Z"} executionInfo={"elapsed": 7698, "status": "ok", "timestamp": 1602557854982, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="s_ceUlxOKg7V" output_cleared=false
df['Brand'] = df['Name'].apply(lambda x: x.split(' ')[0])
df['Series'] = df['Name'].apply(lambda x: x.split(' ')[1])
df['Type'] = df['Name'].apply(lambda x: x.split(' ')[2])
df.drop(columns='Name', inplace=True)

# %% cell_id="00011-a1625de7-7e59-420d-844b-5a3a2d1e652d" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-15T12:54:13.170603Z", "iopub.status.busy": "2020-10-15T12:54:13.170603Z", "iopub.status.idle": "2020-10-15T12:54:13.194464Z", "shell.execute_reply": "2020-10-15T12:54:13.193466Z", "shell.execute_reply.started": "2020-10-15T12:54:13.170603Z"} executionInfo={"elapsed": 7675, "status": "ok", "timestamp": 1602557854983, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="aY7bafplKu4K" outputId="377dfcdc-b00b-464e-b9e1-bf25f058da5d" output_cleared=false
df.head()

# %% cell_id="00012-d9063eee-15ac-4071-9fe3-e39d9dd7cf01" colab={"base_uri": "https://localhost:8080/", "height": 80} execution={"iopub.execute_input": "2020-10-15T12:54:13.198462Z", "iopub.status.busy": "2020-10-15T12:54:13.197460Z", "iopub.status.idle": "2020-10-15T12:54:13.241337Z", "shell.execute_reply": "2020-10-15T12:54:13.240341Z", "shell.execute_reply.started": "2020-10-15T12:54:13.198462Z"} executionInfo={"elapsed": 7659, "status": "ok", "timestamp": 1602557854984, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="VGanyS85IajY" outputId="78b54a45-cd8e-421d-8986-f4425808a80f" output_cleared=false tags=[]
df[(df['Engine']=='72 CC')]

# %% cell_id="00013-4aa8d13b-6dab-4f52-ae70-fc8f25367feb" colab={"base_uri": "https://localhost:8080/", "height": 68} execution={"iopub.execute_input": "2020-10-15T12:54:13.242335Z", "iopub.status.busy": "2020-10-15T12:54:13.242335Z", "iopub.status.idle": "2020-10-15T12:54:13.288213Z", "shell.execute_reply": "2020-10-15T12:54:13.287214Z", "shell.execute_reply.started": "2020-10-15T12:54:13.242335Z"} executionInfo={"elapsed": 7643, "status": "ok", "timestamp": 1602557854985, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="6m3Suen4Z_9Y" outputId="dfa71fa1-e6b4-49d1-cf08-52501b67afe3" output_cleared=false tags=[]
print('Satuan pada feature Mileage:', df['Mileage'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).unique())
print('Satuan pada feature Engine:', df['Engine'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).unique())
print('Satuan pada feature Power:', df['Power'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).unique())

# %% colab={"base_uri": "https://localhost:8080/", "height": 68} execution={"iopub.execute_input": "2020-10-15T12:54:13.289210Z", "iopub.status.busy": "2020-10-15T12:54:13.289210Z", "iopub.status.idle": "2020-10-15T12:54:13.320127Z", "shell.execute_reply": "2020-10-15T12:54:13.319129Z", "shell.execute_reply.started": "2020-10-15T12:54:13.289210Z"} executionInfo={"elapsed": 7625, "status": "ok", "timestamp": 1602557854986, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="IPkNA-lwZIrE" outputId="2336f28d-a97d-40ad-d2a9-0c61d22ca548" tags=[]
print('Invalid Value pada feature Mileage:', pd.Series([x for x in df['Mileage'] if str(x).split(' ')[0].isalpha()]).unique())
print('Invalid Value pada feature Engine:', pd.Series([x for x in df['Engine'] if str(x).split(' ')[0].isalpha()]).unique())
print('Invalid Value pada feature Power:', pd.Series([x for x in df['Power'] if str(x).split(' ')[0].isalpha()]).unique())

# %% cell_id="00014-3095d6e2-739a-4e9a-91dd-1b6a1aeecb0c" colab={"base_uri": "https://localhost:8080/", "height": 68} execution={"iopub.execute_input": "2020-10-15T12:54:13.322124Z", "iopub.status.busy": "2020-10-15T12:54:13.322124Z", "iopub.status.idle": "2020-10-15T12:54:13.352043Z", "shell.execute_reply": "2020-10-15T12:54:13.351046Z", "shell.execute_reply.started": "2020-10-15T12:54:13.322124Z"} executionInfo={"elapsed": 7605, "status": "ok", "timestamp": 1602557854987, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="fvUS80s_t_k8" outputId="08d25d02-34df-4bc3-b2d4-cb4613f9b9c2" output_cleared=false
df['Mileage'].apply(lambda x: x if pd.isna(x) else x.split(' ')[1]).value_counts()

# %% cell_id="00016-3c631710-d459-4dc7-854c-9b601a8f78a7" execution={"iopub.execute_input": "2020-10-15T12:54:13.354037Z", "iopub.status.busy": "2020-10-15T12:54:13.354037Z", "iopub.status.idle": "2020-10-15T12:54:13.415872Z", "shell.execute_reply": "2020-10-15T12:54:13.414874Z", "shell.execute_reply.started": "2020-10-15T12:54:13.354037Z"} executionInfo={"elapsed": 7601, "status": "ok", "timestamp": 1602557854987, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="3kGt8wrVLscx" output_cleared=false
df['Mileage (kmpl)'] = df['Mileage'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])
df['Engine (CC)'] = df['Engine'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])
df['Power (bhp)'] = df['Power'].apply(lambda x: x if pd.isna(x) else x.split(' ')[0])

df['Mileage (kmpl)'] = pd.to_numeric(df['Mileage (kmpl)'], errors='coerce')
df['Engine (CC)'] = pd.to_numeric(df['Engine (CC)'], errors='coerce')
df['Power (bhp)'] = pd.to_numeric(df['Power (bhp)'], errors='coerce')

df.drop(columns=['Mileage', 'Engine', 'Power'], inplace=True)

# %% cell_id="00017-1cd20bd3-e6f2-422f-aceb-75c7b5c48651" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-15T12:54:13.416868Z", "iopub.status.busy": "2020-10-15T12:54:13.416868Z", "iopub.status.idle": "2020-10-15T12:54:13.447785Z", "shell.execute_reply": "2020-10-15T12:54:13.446788Z", "shell.execute_reply.started": "2020-10-15T12:54:13.416868Z"} executionInfo={"elapsed": 7583, "status": "ok", "timestamp": 1602557854988, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="L0OayFyZd12B" outputId="2f48f4c2-7706-4154-cec3-c2049d985c1a" output_cleared=false
df.head()

# %% allow_embed=false cell_id="00018-449a1211-5734-4bdb-a58d-5722fc129da5" colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:54:13.449780Z", "iopub.status.busy": "2020-10-15T12:54:13.448783Z", "iopub.status.idle": "2020-10-15T12:54:13.511429Z", "shell.execute_reply": "2020-10-15T12:54:13.510432Z", "shell.execute_reply.started": "2020-10-15T12:54:13.449780Z"} executionInfo={"elapsed": 7944, "status": "ok", "timestamp": 1602557855365, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="FMBbBCMFIajt" outputId="f9e41389-4aef-42f7-c85b-3ce466d44ad0" output_cleared=false tags=[]
df.describe()

# %% cell_id="00019-2ad88c07-d7d0-4fb4-a7b0-94c417269d0a" colab={"base_uri": "https://localhost:8080/", "height": 419} execution={"iopub.execute_input": "2020-10-15T12:54:13.513423Z", "iopub.status.busy": "2020-10-15T12:54:13.512425Z", "iopub.status.idle": "2020-10-15T12:54:13.561330Z", "shell.execute_reply": "2020-10-15T12:54:13.560333Z", "shell.execute_reply.started": "2020-10-15T12:54:13.513423Z"} executionInfo={"elapsed": 7928, "status": "ok", "timestamp": 1602557855366, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="PJFktTnEIajw" outputId="79dc011e-2a51-469e-adda-f60748d14221" output_cleared=false tags=[]
df[df['Mileage (kmpl)']==0]

# %% cell_id="00020-e43329af-54de-4465-87d6-cb3c4c39a087" colab={"base_uri": "https://localhost:8080/", "height": 80} execution={"iopub.execute_input": "2020-10-15T12:54:13.564326Z", "iopub.status.busy": "2020-10-15T12:54:13.563325Z", "iopub.status.idle": "2020-10-15T12:54:13.592266Z", "shell.execute_reply": "2020-10-15T12:54:13.591270Z", "shell.execute_reply.started": "2020-10-15T12:54:13.564326Z"} executionInfo={"elapsed": 7914, "status": "ok", "timestamp": 1602557855367, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="u6RCnI2OIajz" outputId="4703b6e0-e354-406d-98ea-e3eb349c8c8d" output_cleared=false tags=[]
df[df['Seats']==0]

# %% cell_id="00021-e0395a10-0bb0-43fd-99ef-52126dbb0aa8" execution={"iopub.execute_input": "2020-10-15T12:54:13.593244Z", "iopub.status.busy": "2020-10-15T12:54:13.593244Z", "iopub.status.idle": "2020-10-15T12:54:13.608205Z", "shell.execute_reply": "2020-10-15T12:54:13.607236Z", "shell.execute_reply.started": "2020-10-15T12:54:13.593244Z"} executionInfo={"elapsed": 7912, "status": "ok", "timestamp": 1602557855368, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Afk7mbdfIaj2" output_cleared=false tags=[]
df['Mileage (kmpl)'] = df['Mileage (kmpl)'].replace(0, np.nan)
df['Seats'] = df['Seats'].replace(0, np.nan)

# %% cell_id="00022-bab6e9eb-e370-437d-bee9-837a51ebbb43" colab={"base_uri": "https://localhost:8080/", "height": 153} execution={"iopub.execute_input": "2020-10-15T12:54:13.609203Z", "iopub.status.busy": "2020-10-15T12:54:13.609203Z", "iopub.status.idle": "2020-10-15T12:54:13.639120Z", "shell.execute_reply": "2020-10-15T12:54:13.639120Z", "shell.execute_reply.started": "2020-10-15T12:54:13.609203Z"} executionInfo={"elapsed": 7898, "status": "ok", "timestamp": 1602557855369, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="wmm-4vqkIaj5" outputId="8c8c575e-3e82-45be-a919-f975933e10f1" output_cleared=false tags=[]
cat_cols = [col for col in df.columns if df[col].dtypes == 'object']
df[cat_cols].nunique()

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-15T12:54:13.641117Z", "iopub.status.busy": "2020-10-15T12:54:13.641117Z", "iopub.status.idle": "2020-10-15T12:54:13.656078Z", "shell.execute_reply": "2020-10-15T12:54:13.655080Z", "shell.execute_reply.started": "2020-10-15T12:54:13.641117Z"} executionInfo={"elapsed": 7881, "status": "ok", "timestamp": 1602557855370, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Q2UrO_LJZIri" outputId="d0cdcf98-da27-4b8f-9b0e-c7906e47e546"
['a', 'b']

# %% cell_id="00023-826cb95b-08be-4095-8a0b-489a9f2a0fcc" colab={"base_uri": "https://localhost:8080/", "height": 1000} execution={"iopub.execute_input": "2020-10-15T12:54:13.657075Z", "iopub.status.busy": "2020-10-15T12:54:13.657075Z", "iopub.status.idle": "2020-10-15T12:54:13.686027Z", "shell.execute_reply": "2020-10-15T12:54:13.686027Z", "shell.execute_reply.started": "2020-10-15T12:54:13.657075Z"} executionInfo={"elapsed": 7865, "status": "ok", "timestamp": 1602557855371, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="YLT0EpfVlHZ2" outputId="612b2218-d5f4-41fd-a4f4-5f2c74655110" output_cleared=false tags=[]
for col in cat_cols:
  print(col, df[col].unique(), '\n')

# %% cell_id="00024-47c53a05-57f5-4f76-8ebc-626044400a24" execution={"iopub.execute_input": "2020-10-15T12:54:13.687993Z", "iopub.status.busy": "2020-10-15T12:54:13.687993Z", "iopub.status.idle": "2020-10-15T12:54:13.701957Z", "shell.execute_reply": "2020-10-15T12:54:13.700991Z", "shell.execute_reply.started": "2020-10-15T12:54:13.687993Z"} executionInfo={"elapsed": 7862, "status": "ok", "timestamp": 1602557855372, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Ei1r1wmzlRHu" output_cleared=false
df['Brand'] = df['Brand'].replace('ISUZU', 'Isuzu')

# %% cell_id="00025-57eb8332-9868-4559-adc8-cbaa48039fcd" colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-15T12:54:13.707937Z", "iopub.status.busy": "2020-10-15T12:54:13.706942Z", "iopub.status.idle": "2020-10-15T12:54:13.717912Z", "shell.execute_reply": "2020-10-15T12:54:13.716947Z", "shell.execute_reply.started": "2020-10-15T12:54:13.706942Z"} executionInfo={"elapsed": 7844, "status": "ok", "timestamp": 1602557855372, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="OXBlXSF3mZ4J" outputId="bb0f36dd-fab7-4f4a-fbd2-20ea42a9dd9d" output_cleared=false tags=[]
print('Brand', df['Brand'].unique())

# %% cell_id="00026-3a835b06-20e7-4253-852f-976a27b4f497" colab={"base_uri": "https://localhost:8080/", "height": 119} execution={"iopub.execute_input": "2020-10-15T12:54:13.719939Z", "iopub.status.busy": "2020-10-15T12:54:13.719939Z", "iopub.status.idle": "2020-10-15T12:54:13.749824Z", "shell.execute_reply": "2020-10-15T12:54:13.748862Z", "shell.execute_reply.started": "2020-10-15T12:54:13.719939Z"} executionInfo={"elapsed": 7829, "status": "ok", "timestamp": 1602557855373, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Q8rkmxpdm0Oz" outputId="82012ee0-04a0-43fa-9b89-8b553b9e8224" output_cleared=false
df['Fuel_Type'].value_counts()

# %% cell_id="00027-bbfa9e1f-1505-4f93-8bb1-e55ff6cae887" colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-15T12:54:13.750822Z", "iopub.status.busy": "2020-10-15T12:54:13.750822Z", "iopub.status.idle": "2020-10-15T12:54:13.765785Z", "shell.execute_reply": "2020-10-15T12:54:13.764821Z", "shell.execute_reply.started": "2020-10-15T12:54:13.750822Z"} executionInfo={"elapsed": 7813, "status": "ok", "timestamp": 1602557855374, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="zQVZX-GcIakM" outputId="5cd57e97-d41d-4239-cc33-7daa2f500015" output_cleared=false tags=[]
df['Series'].nunique()

# %% cell_id="00030-25fa031b-975c-43ae-9e06-a422b44e4789" colab={"base_uri": "https://localhost:8080/", "height": 204} execution={"iopub.execute_input": "2020-10-15T12:54:13.766780Z", "iopub.status.busy": "2020-10-15T12:54:13.766780Z", "iopub.status.idle": "2020-10-15T12:54:13.797697Z", "shell.execute_reply": "2020-10-15T12:54:13.796701Z", "shell.execute_reply.started": "2020-10-15T12:54:13.766780Z"} executionInfo={"elapsed": 7798, "status": "ok", "timestamp": 1602557855375, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="89WdS5GnIakU" outputId="6264b415-c8e5-424a-ff4c-f4352c89eb16" output_cleared=false tags=[]
df.head()

# %% cell_id="00031-7ddc6533-c13e-4779-b1f0-09774747357f" colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:54:13.798694Z", "iopub.status.busy": "2020-10-15T12:54:13.798694Z", "iopub.status.idle": "2020-10-15T12:54:13.845569Z", "shell.execute_reply": "2020-10-15T12:54:13.844623Z", "shell.execute_reply.started": "2020-10-15T12:54:13.798694Z"} executionInfo={"elapsed": 8308, "status": "ok", "timestamp": 1602557855905, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Ney4gM1WIakY" outputId="36e479a6-a706-4b51-c4af-0814c7c27cb2" output_cleared=false tags=[]
df.describe()

# %% cell_id="00032-a59a0124-6b41-4b36-aead-c0d4df135d9b" colab={"base_uri": "https://localhost:8080/", "height": 173} execution={"iopub.execute_input": "2020-10-15T12:54:13.849559Z", "iopub.status.busy": "2020-10-15T12:54:13.848562Z", "iopub.status.idle": "2020-10-15T12:54:13.909398Z", "shell.execute_reply": "2020-10-15T12:54:13.908402Z", "shell.execute_reply.started": "2020-10-15T12:54:13.849559Z"} executionInfo={"elapsed": 8286, "status": "ok", "timestamp": 1602557855906, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="3Jk-RIU6Iakb" outputId="7a8d1a40-3f4b-4597-ca7b-047f72aa0f7c" output_cleared=false tags=[]
df.describe(include=['object']) 

# %% colab={"base_uri": "https://localhost:8080/", "height": 483} execution={"iopub.execute_input": "2020-10-15T12:54:13.912389Z", "iopub.status.busy": "2020-10-15T12:54:13.911395Z", "iopub.status.idle": "2020-10-15T12:54:13.941312Z", "shell.execute_reply": "2020-10-15T12:54:13.940315Z", "shell.execute_reply.started": "2020-10-15T12:54:13.912389Z"} executionInfo={"elapsed": 8268, "status": "ok", "timestamp": 1602557855907, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="PMAtruMZZIsX" outputId="8c07d1b9-8321-47bd-d866-8ff2127b58fc"
explor.null_checker(df)

# %% cell_id="00034-a74bd3e4-e566-4425-a0a5-cc1c9af4f640" execution={"iopub.execute_input": "2020-10-15T12:54:13.942312Z", "iopub.status.busy": "2020-10-15T12:54:13.942312Z", "iopub.status.idle": "2020-10-15T12:54:13.956273Z", "shell.execute_reply": "2020-10-15T12:54:13.955275Z", "shell.execute_reply.started": "2020-10-15T12:54:13.942312Z"} executionInfo={"elapsed": 8266, "status": "ok", "timestamp": 1602557855908, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="LbW9gLQVIakg" output_cleared=false tags=[]
df.loc[df['Fuel_Type']=='Electric', 'Mileage (kmpl)'] = df.loc[df['Fuel_Type']=='Electric', 'Mileage (kmpl)'].replace(np.nan, 0)

# %% cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" colab={"base_uri": "https://localhost:8080/", "height": 111} execution={"iopub.execute_input": "2020-10-15T12:54:13.957271Z", "iopub.status.busy": "2020-10-15T12:54:13.957271Z", "iopub.status.idle": "2020-10-15T12:54:13.986194Z", "shell.execute_reply": "2020-10-15T12:54:13.985196Z", "shell.execute_reply.started": "2020-10-15T12:54:13.957271Z"} executionInfo={"elapsed": 8251, "status": "ok", "timestamp": 1602557855908, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="bKob_zWgIakl" outputId="baf516c7-97ef-40f1-a137-4a3c21b0e9dc" output_cleared=false tags=[]
df.loc[df['Fuel_Type']=='Electric']

# %% [markdown] id="g1GS1AAUZIt9"
# # Preprocessing

# %% colab={"base_uri": "https://localhost:8080/", "height": 34} execution={"iopub.execute_input": "2020-10-15T12:54:13.988187Z", "iopub.status.busy": "2020-10-15T12:54:13.987190Z", "iopub.status.idle": "2020-10-15T12:54:14.002152Z", "shell.execute_reply": "2020-10-15T12:54:14.001155Z", "shell.execute_reply.started": "2020-10-15T12:54:13.988187Z"} executionInfo={"elapsed": 8249, "status": "ok", "timestamp": 1602557855909, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="INV8VvOYZItN" outputId="d4ca600b-dc4f-4db2-fba8-54aa6678f0c0"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape

# %% [markdown] id="yEgVyyNSZIt9"
# ## Train test split

# %% execution={"iopub.execute_input": "2020-10-15T12:54:14.005145Z", "iopub.status.busy": "2020-10-15T12:54:14.004145Z", "iopub.status.idle": "2020-10-15T12:54:14.033068Z", "shell.execute_reply": "2020-10-15T12:54:14.032071Z", "shell.execute_reply.started": "2020-10-15T12:54:14.005145Z"} executionInfo={"elapsed": 8232, "status": "ok", "timestamp": 1602557855911, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="nPxFt6bSZIt-"
# melakukan train test split di awal untuk mencegah data bocor ke test set saat dilakukan encoding/imputation
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

# %% [markdown] id="oxqsMHrKZIuA"
# ## Encoding

# %% cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-15T12:54:14.034066Z", "iopub.status.busy": "2020-10-15T12:54:14.034066Z", "iopub.status.idle": "2020-10-15T12:54:14.187654Z", "shell.execute_reply": "2020-10-15T12:54:14.186657Z", "shell.execute_reply.started": "2020-10-15T12:54:14.034066Z"} executionInfo={"elapsed": 1054, "status": "ok", "timestamp": 1602557861674, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="_0criLnZIakn" outputId="766c5c78-5fac-492b-e39c-674c73139932" output_cleared=false tags=[]
# One hot encoding
col_to_encode = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
oh_encoder = ce.OneHotEncoder(cols=col_to_encode,
                              use_cat_names=True)
oh_encoder.fit(X_train)

# Encoding train set
X_train = oh_encoder.transform(X_train)
# Encoding test set
X_test = oh_encoder.transform(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-15T12:54:14.188652Z", "iopub.status.busy": "2020-10-15T12:54:14.188652Z", "iopub.status.idle": "2020-10-15T12:54:14.267439Z", "shell.execute_reply": "2020-10-15T12:54:14.266443Z", "shell.execute_reply.started": "2020-10-15T12:54:14.188652Z"} executionInfo={"elapsed": 587, "status": "ok", "timestamp": 1602557861677, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="kcMLnvJxZIuD" outputId="ec506ea3-a38a-4b80-9e62-2f3af531162a"
# Target encoding/One hot encoding untuk feature dengan kategori yang banyak
col_to_encode = ['Series', 'Type']
target_encoder = ce.TargetEncoder(cols=col_to_encode)
target_encoder.fit(X_train, y_train)

# Encoding train set
X_train = target_encoder.transform(X_train)
# Encoding test set
X_test = target_encoder.transform(X_test)

# %% [markdown] id="6MJs1hK7Iv1N"
# ## Missing Value Imputation

# %% execution={"iopub.execute_input": "2020-10-15T12:54:14.269434Z", "iopub.status.busy": "2020-10-15T12:54:14.269434Z", "iopub.status.idle": "2020-10-15T12:54:25.487160Z", "shell.execute_reply": "2020-10-15T12:54:25.487160Z", "shell.execute_reply.started": "2020-10-15T12:54:14.269434Z"} executionInfo={"elapsed": 9747, "status": "ok", "timestamp": 1602558096090, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ccgkETh_Iv1O"
# memprediksi nilai missing value dengan algoritma 
imputer = mf.KernelDataSet(
  X_train,
  save_all_iterations=True,
  random_state=1991,
  mean_match_candidates=5
)

imputer.mice(10)

# %% execution={"iopub.execute_input": "2020-10-15T12:54:25.487160Z", "iopub.status.busy": "2020-10-15T12:54:25.487160Z", "iopub.status.idle": "2020-10-15T12:54:25.503631Z", "shell.execute_reply": "2020-10-15T12:54:25.502669Z", "shell.execute_reply.started": "2020-10-15T12:54:25.487160Z"} executionInfo={"elapsed": 769, "status": "ok", "timestamp": 1602558116061, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="e_zrbZk6Iv1S"
# Train set imputation
X_train_full = imputer.complete_data()

# %% execution={"iopub.execute_input": "2020-10-15T12:54:25.505624Z", "iopub.status.busy": "2020-10-15T12:54:25.504627Z", "iopub.status.idle": "2020-10-15T12:54:27.936064Z", "shell.execute_reply": "2020-10-15T12:54:27.936064Z", "shell.execute_reply.started": "2020-10-15T12:54:25.505624Z"} executionInfo={"elapsed": 2626, "status": "ok", "timestamp": 1602558147720, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="s3TrxVjQIv1Z"
# Test set imputation
new_data = imputer.impute_new_data(X_test)
X_test_full = new_data.complete_data()

# %% [markdown] id="zYZVKzQYIxVx"
# ## Feature Selection

# %% execution={"iopub.execute_input": "2020-10-15T12:54:27.936064Z", "iopub.status.busy": "2020-10-15T12:54:27.936064Z", "iopub.status.idle": "2020-10-15T12:54:28.031986Z", "shell.execute_reply": "2020-10-15T12:54:28.030987Z", "shell.execute_reply.started": "2020-10-15T12:54:27.936064Z"} executionInfo={"elapsed": 974, "status": "ok", "timestamp": 1602558988123, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="5RY-BaL8IxVy"
# Memfilter feature dengan korelasi tinggi
corr_price = X_train.join(y_train).corr()['Price']
index = corr_price[(corr_price < -0.20) | (corr_price > 0.20)].index

X_train_full =  X_train_full[index[:-1]]
X_test_full = X_test_full[index[:-1]]


# %% [markdown] id="wV2sjkqEZIup"
# # Modeling

# %% [markdown] id="4g_nWqotKl6_"
# ## Functions

# %% execution={"iopub.execute_input": "2020-10-15T12:54:28.032983Z", "iopub.status.busy": "2020-10-15T12:54:28.032983Z", "iopub.status.idle": "2020-10-15T12:54:28.047942Z", "shell.execute_reply": "2020-10-15T12:54:28.046978Z", "shell.execute_reply.started": "2020-10-15T12:54:28.032983Z"} executionInfo={"elapsed": 1009, "status": "ok", "timestamp": 1602559132744, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Qp4QHIuFZIuq"
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


# %% execution={"iopub.execute_input": "2020-10-15T12:54:28.049965Z", "iopub.status.busy": "2020-10-15T12:54:28.048938Z", "iopub.status.idle": "2020-10-15T12:54:28.063899Z", "shell.execute_reply": "2020-10-15T12:54:28.061903Z", "shell.execute_reply.started": "2020-10-15T12:54:28.049965Z"} executionInfo={"elapsed": 806, "status": "ok", "timestamp": 1602559132746, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="BXEr8F5VZIu0"
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


# %% [markdown] id="aR4Sp3UCZIu2"
# ## Base Model

# %% execution={"iopub.execute_input": "2020-10-15T12:54:28.065895Z", "iopub.status.busy": "2020-10-15T12:54:28.065895Z", "iopub.status.idle": "2020-10-15T12:54:28.079856Z", "shell.execute_reply": "2020-10-15T12:54:28.077864Z", "shell.execute_reply.started": "2020-10-15T12:54:28.065895Z"} executionInfo={"elapsed": 678, "status": "ok", "timestamp": 1602559134050, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Oux2OxeDZIu2"
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

# %% [markdown] id="kCSEOF35MoSB"
# ### Unscaled dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:54:28.081881Z", "iopub.status.busy": "2020-10-15T12:54:28.080854Z", "iopub.status.idle": "2020-10-15T12:55:11.188310Z", "shell.execute_reply": "2020-10-15T12:55:11.187312Z", "shell.execute_reply.started": "2020-10-15T12:54:28.081881Z"} executionInfo={"elapsed": 30383, "status": "ok", "timestamp": 1602559165523, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="857f512d-6910-4625-e01b-c6b587a9094c"
# evaluasi model memakai function
unscaled = evaluate_model(models, X_train_full, X_test_full, y_train, y_test)

# %% [markdown] id="AodaQJBNMtob"
# ### Scaled dataset

# %% execution={"iopub.execute_input": "2020-10-15T12:55:11.191302Z", "iopub.status.busy": "2020-10-15T12:55:11.190305Z", "iopub.status.idle": "2020-10-15T12:55:11.236183Z", "shell.execute_reply": "2020-10-15T12:55:11.235184Z", "shell.execute_reply.started": "2020-10-15T12:55:11.191302Z"} executionInfo={"elapsed": 25276, "status": "ok", "timestamp": 1602559165525, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="2lQZQbORMwYB"
# Scaling data
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train_full)
X_train_full_scaled = scaler.transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:55:11.239174Z", "iopub.status.busy": "2020-10-15T12:55:11.238177Z", "iopub.status.idle": "2020-10-15T12:55:54.767071Z", "shell.execute_reply": "2020-10-15T12:55:54.767071Z", "shell.execute_reply.started": "2020-10-15T12:55:11.239174Z"} executionInfo={"elapsed": 54513, "status": "ok", "timestamp": 1602559195270, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="06962bd1-1bb2-4c3e-bd1c-74e71a1d0ed5"
# evaluasi model memakai function
scaled = evaluate_model(models, X_train_full_scaled, X_test_full_scaled, y_train, y_test)

# %% [markdown] id="bg_vcQxLLg0n"
# ### Summarizing

# %% execution={"iopub.execute_input": "2020-10-15T12:55:54.771050Z", "iopub.status.busy": "2020-10-15T12:55:54.770053Z", "iopub.status.idle": "2020-10-15T12:55:54.784016Z", "shell.execute_reply": "2020-10-15T12:55:54.783018Z", "shell.execute_reply.started": "2020-10-15T12:55:54.771050Z"}
unscaled['Dataset Version'] = 'imputed + selected + unscaled'
scaled['Dataset Version'] = 'imputed + selected + scaled'

# %% execution={"iopub.execute_input": "2020-10-15T12:55:54.786011Z", "iopub.status.busy": "2020-10-15T12:55:54.786011Z", "iopub.status.idle": "2020-10-15T12:55:54.831887Z", "shell.execute_reply": "2020-10-15T12:55:54.830889Z", "shell.execute_reply.started": "2020-10-15T12:55:54.786011Z"}
imputed_selected = pd.concat([unscaled, scaled], axis=0)
imputed_selected

# %% execution={"iopub.execute_input": "2020-10-15T12:55:54.834878Z", "iopub.status.busy": "2020-10-15T12:55:54.833882Z", "iopub.status.idle": "2020-10-15T12:55:54.847844Z", "shell.execute_reply": "2020-10-15T12:55:54.846847Z", "shell.execute_reply.started": "2020-10-15T12:55:54.834878Z"}
imputed_selected.to_csv('imputed_selected.csv')
