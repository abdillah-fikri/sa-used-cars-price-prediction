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
#         hash: 8d4d772f21767a3a72f3356b4ab1badff3b831eb21eba306d4ebdf1fe7777d12
#     name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
# ---

# %% [markdown]
# # Data Importing

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# %%
df = pd.read_csv('../data/processed/after_prep.csv')
df.head()

# %%
df.info()

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Outlier Handling

# %%
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape


# %% [markdown]
# ## Feature Enginering

# %%
# Create price zone feature from Location
def zone(data):
    if data in ["Kolkata"]:
        return "Eastern"
    elif data in ["Delhi", "Jaipur"]:
        return "Northern"
    elif data in ["Ahmedabad", "Mumbai", "Pune"]:
        return "Western"
    else:
        return "Southern"

df["Zone"] = df["Location"].apply(zone)

# %%
df.info()

# %% [markdown]
# ## Train test split

# %%
# melakukan train test split di awal untuk mencegah data bocor ke test set saat dilakukan encoding/imputation
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)

# %% [markdown]
# ## Encoding

# %%
# Encoding categorical features
col_to_encode = df.select_dtypes("object").columns.tolist()
encoder = ce.CatBoostEncoder(cols=col_to_encode)
encoder.fit(X_train, y_train)

# Encoding train set
X_train = encoder.transform(X_train)
# Encoding test set
X_test = encoder.transform(X_test)

# %% [markdown]
# ## Missing Value Imputation

# %%
# memprediksi nilai missing value dengan algoritma 
imputer = mf.KernelDataSet(
  X_train,
  save_all_iterations=True,
  random_state=1991,
  mean_match_candidates=5
)
imputer.mice(10)

# %%
# Train set imputation
X_train_full = imputer.complete_data()

# %%
# Test set imputation
new_data = imputer.impute_new_data(X_test)
X_test_full = new_data.complete_data()


# %% [markdown]
# # Modeling

# %% [markdown]
# ## Functions

# %%
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


# %%
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


# %% [markdown]
# ## Base Models

# %%
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

# %%
evaluate_model(models, X_train_full, X_test_full, y_train, y_test)

# %% [markdown]
# ### Feature Importance

# %%
xgb_model.fit(X_train_full, y_train)

# %%
feat_imp = pd.DataFrame(xgb_model.feature_importances_, index=X_train_full.columns)
feat_imp

# %%
xgb.plot_importance(xgb_model)

# %%
lgb.plot_importance(lgb_model)

# %%
lgb_model.feature_importances_

# %%
