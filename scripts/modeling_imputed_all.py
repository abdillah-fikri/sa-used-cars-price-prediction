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

# %% execution={"iopub.execute_input": "2020-10-15T12:50:17.869621Z", "iopub.status.busy": "2020-10-15T12:50:17.869621Z", "iopub.status.idle": "2020-10-15T12:50:17.883583Z", "shell.execute_reply": "2020-10-15T12:50:17.882586Z", "shell.execute_reply.started": "2020-10-15T12:50:17.869621Z"} id="INV8VvOYZItN"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape

# %% [markdown] id="yEgVyyNSZIt9"
# ## Train test split

# %% execution={"iopub.execute_input": "2020-10-15T12:50:17.884579Z", "iopub.status.busy": "2020-10-15T12:50:17.884579Z", "iopub.status.idle": "2020-10-15T12:50:17.898543Z", "shell.execute_reply": "2020-10-15T12:50:17.897546Z", "shell.execute_reply.started": "2020-10-15T12:50:17.884579Z"} id="nPxFt6bSZIt-" outputId="2b131b44-7d5e-469d-9e5f-0bc241abd283"
# melakukan train test split di awal untuk mencegah data bocor ke test set saat dilakukan encoding/imputation
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

# %% [markdown] id="oxqsMHrKZIuA"
# ## Encoding

# %% cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" execution={"iopub.execute_input": "2020-10-15T12:50:17.900538Z", "iopub.status.busy": "2020-10-15T12:50:17.899538Z", "iopub.status.idle": "2020-10-15T12:50:18.100999Z", "shell.execute_reply": "2020-10-15T12:50:18.100001Z", "shell.execute_reply.started": "2020-10-15T12:50:17.900538Z"} id="_0criLnZIakn" output_cleared=false tags=[]
# # One hot encoding
# col_to_encode = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']
# oh_encoder = ce.OneHotEncoder(cols=col_to_encode,
#                               use_cat_names=True)
# oh_encoder.fit(X_train)

# # Encoding train set
# X_train = oh_encoder.transform(X_train)
# # Encoding test set
# X_test = oh_encoder.transform(X_test)

# %% execution={"iopub.execute_input": "2020-10-15T12:50:18.102994Z", "iopub.status.busy": "2020-10-15T12:50:18.101997Z", "iopub.status.idle": "2020-10-15T12:50:18.179789Z", "shell.execute_reply": "2020-10-15T12:50:18.178825Z", "shell.execute_reply.started": "2020-10-15T12:50:18.102994Z"} id="kcMLnvJxZIuD"
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

# %% execution={"iopub.execute_input": "2020-10-15T12:50:18.181784Z", "iopub.status.busy": "2020-10-15T12:50:18.180785Z", "iopub.status.idle": "2020-10-15T12:50:29.221721Z", "shell.execute_reply": "2020-10-15T12:50:29.221721Z", "shell.execute_reply.started": "2020-10-15T12:50:18.181784Z"} id="ccgkETh_Iv1O"
# memprediksi nilai missing value dengan algoritma 
imputer = mf.KernelDataSet(
  X_train,
  save_all_iterations=True,
  random_state=1991,
  mean_match_candidates=5
)

imputer.mice(10)

# %% execution={"iopub.execute_input": "2020-10-15T12:50:29.221721Z", "iopub.status.busy": "2020-10-15T12:50:29.221721Z", "iopub.status.idle": "2020-10-15T12:50:29.238494Z", "shell.execute_reply": "2020-10-15T12:50:29.237530Z", "shell.execute_reply.started": "2020-10-15T12:50:29.221721Z"} id="e_zrbZk6Iv1S"
# Train set imputation
X_train_full = imputer.complete_data()

# %% execution={"iopub.execute_input": "2020-10-15T12:50:29.239490Z", "iopub.status.busy": "2020-10-15T12:50:29.239490Z", "iopub.status.idle": "2020-10-15T12:50:31.718460Z", "shell.execute_reply": "2020-10-15T12:50:31.718460Z", "shell.execute_reply.started": "2020-10-15T12:50:29.239490Z"} id="s3TrxVjQIv1Z"
# Test set imputation
new_data = imputer.impute_new_data(X_test)
X_test_full = new_data.complete_data()


# %% [markdown] id="wV2sjkqEZIup"
# # Modeling

# %% [markdown] id="4g_nWqotKl6_"
# ## Functions

# %% execution={"iopub.execute_input": "2020-10-15T12:50:31.718460Z", "iopub.status.busy": "2020-10-15T12:50:31.718460Z", "iopub.status.idle": "2020-10-15T12:50:31.734258Z", "shell.execute_reply": "2020-10-15T12:50:31.733261Z", "shell.execute_reply.started": "2020-10-15T12:50:31.718460Z"} id="Qp4QHIuFZIuq"
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


# %% execution={"iopub.execute_input": "2020-10-15T12:50:31.735253Z", "iopub.status.busy": "2020-10-15T12:50:31.735253Z", "iopub.status.idle": "2020-10-15T12:50:31.750216Z", "shell.execute_reply": "2020-10-15T12:50:31.749219Z", "shell.execute_reply.started": "2020-10-15T12:50:31.735253Z"} id="BXEr8F5VZIu0"
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
    
    return round(summary.sort_values(by='CV RMSE'), 4)


# %% [markdown] id="aR4Sp3UCZIu2"
# ## Base Model

# %% execution={"iopub.execute_input": "2020-10-15T12:50:31.751211Z", "iopub.status.busy": "2020-10-15T12:50:31.751211Z", "iopub.status.idle": "2020-10-15T12:50:31.765176Z", "shell.execute_reply": "2020-10-15T12:50:31.764178Z", "shell.execute_reply.started": "2020-10-15T12:50:31.751211Z"} id="Oux2OxeDZIu2"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:50:31.768167Z", "iopub.status.busy": "2020-10-15T12:50:31.767170Z", "iopub.status.idle": "2020-10-15T12:51:28.813751Z", "shell.execute_reply": "2020-10-15T12:51:28.812752Z", "shell.execute_reply.started": "2020-10-15T12:50:31.768167Z"} executionInfo={"elapsed": 38364, "status": "ok", "timestamp": 1602353945658, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="DgfsmUm-HqGG" outputId="890d1059-fe50-4ed7-87d9-16413c775534"
# evaluasi model memakai function
unscaled = evaluate_model(models, X_train_full, X_test_full, y_train, y_test)

# %% [markdown] id="AodaQJBNMtob"
# ### Scaled dataset

# %% execution={"iopub.execute_input": "2020-10-15T12:51:28.815746Z", "iopub.status.busy": "2020-10-15T12:51:28.815746Z", "iopub.status.idle": "2020-10-15T12:51:28.893537Z", "shell.execute_reply": "2020-10-15T12:51:28.892540Z", "shell.execute_reply.started": "2020-10-15T12:51:28.815746Z"} id="2lQZQbORMwYB"
# Scaling data
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train_full)
X_train_full_scaled = scaler.transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)

# %% colab={"base_uri": "https://localhost:8080/", "height": 297} execution={"iopub.execute_input": "2020-10-15T12:51:28.896529Z", "iopub.status.busy": "2020-10-15T12:51:28.895532Z", "iopub.status.idle": "2020-10-15T12:52:30.847826Z", "shell.execute_reply": "2020-10-15T12:52:30.847826Z", "shell.execute_reply.started": "2020-10-15T12:51:28.896529Z"} executionInfo={"elapsed": 81010, "status": "ok", "timestamp": 1602353988430, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="58C87fQHNRII" outputId="90af7df3-a745-4722-d77d-53f144212a91"
# evaluasi model memakai function
scaled = evaluate_model(models, X_train_full_scaled, X_test_full_scaled, y_train, y_test)

# %% [markdown]
# ### Summarizing

# %% execution={"iopub.execute_input": "2020-10-15T12:52:30.847826Z", "iopub.status.busy": "2020-10-15T12:52:30.847826Z", "iopub.status.idle": "2020-10-15T12:52:30.863806Z", "shell.execute_reply": "2020-10-15T12:52:30.862807Z", "shell.execute_reply.started": "2020-10-15T12:52:30.847826Z"} id="bg_vcQxLLg0n"
unscaled['Dataset Version'] = 'imputed + all + unscaled'
scaled['Dataset Version'] = 'imputed + all + scaled'

# %% execution={"iopub.execute_input": "2020-10-15T12:52:30.865799Z", "iopub.status.busy": "2020-10-15T12:52:30.865799Z", "iopub.status.idle": "2020-10-15T12:52:30.911677Z", "shell.execute_reply": "2020-10-15T12:52:30.910679Z", "shell.execute_reply.started": "2020-10-15T12:52:30.865799Z"}
imputed_all = pd.concat([unscaled, scaled], axis=0)
imputed_all

# %% execution={"iopub.execute_input": "2020-10-15T12:52:30.916663Z", "iopub.status.busy": "2020-10-15T12:52:30.916663Z", "iopub.status.idle": "2020-10-15T12:52:30.943590Z", "shell.execute_reply": "2020-10-15T12:52:30.942594Z", "shell.execute_reply.started": "2020-10-15T12:52:30.916663Z"}
imputed_all.to_csv('../data/processed/summary_imputed_all.csv')

# %%
model = LGBMRegressor()
model.fit(X_train_full, y_train)
lgb.plot_importance(model, figsize=(9,6), dpi=500)

# %%
lgb.plot_tree(model, figsize=(8*4,6*4), dpi=300, )
