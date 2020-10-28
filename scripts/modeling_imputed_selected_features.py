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
# # Data Importing

# %%
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

# %%
df = pd.read_csv('../data/processed/after_prep.csv')
df.head()

# %%
df.info()

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
# # One hot encoding
# col_to_encode = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
# oh_encoder = ce.OneHotEncoder(cols=col_to_encode,
#                               use_cat_names=True)
# oh_encoder.fit(X_train)

# # Encoding train set
# X_train = oh_encoder.transform(X_train)
# # Encoding test set
# X_test = oh_encoder.transform(X_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 85} execution={"iopub.execute_input": "2020-10-15T12:54:14.188652Z", "iopub.status.busy": "2020-10-15T12:54:14.188652Z", "iopub.status.idle": "2020-10-15T12:54:14.267439Z", "shell.execute_reply": "2020-10-15T12:54:14.266443Z", "shell.execute_reply.started": "2020-10-15T12:54:14.188652Z"} executionInfo={"elapsed": 587, "status": "ok", "timestamp": 1602557861677, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="kcMLnvJxZIuD" outputId="ec506ea3-a38a-4b80-9e62-2f3af531162a"
# Target encoding
col_to_encode = ['Series', 'Type', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
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
imputed_selected.to_csv('../data/processed/summary_imputed_selected.csv')
