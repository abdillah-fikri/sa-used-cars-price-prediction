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

```python id="kqlZgC2QIUa_" outputId="e8398ccf-4876-4da6-9b79-dacd33e1d5f6" colab={"base_uri": "https://localhost:8080/", "height": 1000}
pip install -r requirements.txt
```

<!-- #region id="Y0sQL3fRHlWV" -->
# Data Importing
<!-- #endregion -->

```python id="HGU4TdhJHlWW" outputId="e2520e58-b5bd-4c3f-cd92-093651b7e452" colab={"base_uri": "https://localhost:8080/"}
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

```python id="PLDflXqmHlWb" outputId="66d30453-246e-4392-deb7-6001ad1424cd" colab={"base_uri": "https://localhost:8080/", "height": 204}
df = pd.read_csv('after_prep.csv')
df.head()
```

```python id="7yBxceHkHlWf" outputId="76b1ec78-3398-4c73-a0c9-7d5e945244b5" colab={"base_uri": "https://localhost:8080/"}
df.info()
```

<!-- #region id="g1GS1AAUZIt9" -->
# Preprocessing
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:50:17.869621Z", "iopub.status.busy": "2020-10-15T12:50:17.869621Z", "iopub.status.idle": "2020-10-15T12:50:17.883583Z", "shell.execute_reply": "2020-10-15T12:50:17.882586Z", "shell.execute_reply.started": "2020-10-15T12:50:17.869621Z"} id="INV8VvOYZItN" outputId="3a63edac-8314-4e12-bb2e-6d05b83a85e9" colab={"base_uri": "https://localhost:8080/"}
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
```

<!-- #region id="Fvvwh2uSWvUU" -->
## Feature enginering
<!-- #endregion -->

```python id="2HEc8zicU0uy"
# from itertools import combinations
# cat_cols = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']

# for col in combinations(cat_cols, 2):
#     df[col[0]+'_'+col[1]] = df[col[0]] + "_" + df[col[1]]
    
# df.head()
```

```python id="SvdzWhppONbl" outputId="1c9a5f8c-f338-43a0-adbf-dab27b161c05" colab={"base_uri": "https://localhost:8080/", "height": 241}
# Make categorical feature interactions
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

object_cols = df.select_dtypes("object").columns
low_cardinality_cols = [col for col in object_cols if df[col].nunique() < 15]
low_cardinality_cols.append('Brand')
interactions = pd.DataFrame(index=df.index)

for features in combinations(low_cardinality_cols,2):
    
    new_interaction = df[features[0]].map(str)+"_"+df[features[1]].map(str)
    
    # encoder = LabelEncoder()
    interactions["_".join(features)] = new_interaction

df = df.join(interactions)
df.head()
```

<!-- #region id="yEgVyyNSZIt9" -->
## Train test split
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:50:17.884579Z", "iopub.status.busy": "2020-10-15T12:50:17.884579Z", "iopub.status.idle": "2020-10-15T12:50:17.898543Z", "shell.execute_reply": "2020-10-15T12:50:17.897546Z", "shell.execute_reply.started": "2020-10-15T12:50:17.884579Z"} id="nPxFt6bSZIt-"
# melakukan train test split di awal auntuk mencegah data bocor ke test set saat dilakukan encoding/imputation
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" execution={"iopub.execute_input": "2020-10-15T12:50:17.900538Z", "iopub.status.busy": "2020-10-15T12:50:17.899538Z", "iopub.status.idle": "2020-10-15T12:50:18.100999Z", "shell.execute_reply": "2020-10-15T12:50:18.100001Z", "shell.execute_reply.started": "2020-10-15T12:50:17.900538Z"} id="_0criLnZIakn" output_cleared=false tags=[]
# # Define category mapping for label encoding
# owner_map = {
#     'First': 1, 
#     'Second': 2, 
#     'Third': 3, 
#     'Fourth & Above': 4
# }
# # Encoding train set
# X_train["Owner_Type"] = X_train["Owner_Type"].map(owner_map)
# # Encoding test set
# X_test["Owner_Type"] = X_test["Owner_Type"].map(owner_map)
```

```python id="EfuIKvdGpL0B" outputId="330a95d5-fe4a-4340-838b-c98539b0009e" colab={"base_uri": "https://localhost:8080/"}
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

```python execution={"iopub.execute_input": "2020-10-15T12:50:18.102994Z", "iopub.status.busy": "2020-10-15T12:50:18.101997Z", "iopub.status.idle": "2020-10-15T12:50:18.179789Z", "shell.execute_reply": "2020-10-15T12:50:18.178825Z", "shell.execute_reply.started": "2020-10-15T12:50:18.102994Z"} id="kcMLnvJxZIuD" outputId="e265bb11-4ada-4276-afbf-831dd3ea89d2" colab={"base_uri": "https://localhost:8080/"}
# Target encoding
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

```python execution={"iopub.execute_input": "2020-10-15T12:50:18.181784Z", "iopub.status.busy": "2020-10-15T12:50:18.180785Z", "iopub.status.idle": "2020-10-15T12:50:29.221721Z", "shell.execute_reply": "2020-10-15T12:50:29.221721Z", "shell.execute_reply.started": "2020-10-15T12:50:18.181784Z"} id="ccgkETh_Iv1O"
# memprediksi nilai missing value dengan algoritma 
imputer = mf.KernelDataSet(
  X_train,
  save_all_iterations=True,
  random_state=1991,
  mean_match_candidates=5
)
imputer.mice(10)
```

```python execution={"iopub.execute_input": "2020-10-15T12:50:29.221721Z", "iopub.status.busy": "2020-10-15T12:50:29.221721Z", "iopub.status.idle": "2020-10-15T12:50:29.238494Z", "shell.execute_reply": "2020-10-15T12:50:29.237530Z", "shell.execute_reply.started": "2020-10-15T12:50:29.221721Z"} id="e_zrbZk6Iv1S"
# Train set imputation
X_train_full = imputer.complete_data()
```

```python execution={"iopub.execute_input": "2020-10-15T12:50:29.239490Z", "iopub.status.busy": "2020-10-15T12:50:29.239490Z", "iopub.status.idle": "2020-10-15T12:50:31.718460Z", "shell.execute_reply": "2020-10-15T12:50:31.718460Z", "shell.execute_reply.started": "2020-10-15T12:50:29.239490Z"} id="s3TrxVjQIv1Z"
# Test set imputation
new_data = imputer.impute_new_data(X_test)
X_test_full = new_data.complete_data()
```

<!-- #region id="wV2sjkqEZIup" -->
# Modeling
<!-- #endregion -->

<!-- #region id="4g_nWqotKl6_" -->
## Functions
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-15T12:50:31.718460Z", "iopub.status.busy": "2020-10-15T12:50:31.718460Z", "iopub.status.idle": "2020-10-15T12:50:31.734258Z", "shell.execute_reply": "2020-10-15T12:50:31.733261Z", "shell.execute_reply.started": "2020-10-15T12:50:31.718460Z"} id="Qp4QHIuFZIuq"
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

```python execution={"iopub.execute_input": "2020-10-15T12:50:31.735253Z", "iopub.status.busy": "2020-10-15T12:50:31.735253Z", "iopub.status.idle": "2020-10-15T12:50:31.750216Z", "shell.execute_reply": "2020-10-15T12:50:31.749219Z", "shell.execute_reply.started": "2020-10-15T12:50:31.735253Z"} id="BXEr8F5VZIu0"
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
    summary.reset_index(inplace=True)
    summary = summary[['Train R2', 'CV R2', 'Test R2',
                       'Train RMSE', 'CV RMSE', 'Test RMSE',
                       'Train MAE', 'CV MAE', 'Test MAE', 'Model']]
    
    return round(summary.sort_values(by='CV RMSE'), 4)
```

<!-- #region id="UR-TUfLrHlUy" -->
## Hyperparameter Tuning
<!-- #endregion -->

```python id="1V7znCDAVRXl" outputId="4f634202-ad44-4606-c9f5-d90c41869243" colab={"base_uri": "https://localhost:8080/", "height": 241}
X_train_full.head()
```

<!-- #region id="kGwkRRZ8W43D" -->
### XGBoost
<!-- #endregion -->

<!-- #region id="PzcpvsXbjHs3" -->
#### Study 1
<!-- #endregion -->

```python id="qwWoTc--ImFJ" outputId="4dc410d4-da23-4a1d-d3ee-d09df8633c24" colab={"base_uri": "https://localhost:8080/"}
 def objective(trial):

    dtrain = xgb.DMatrix(X_train_full, label=y_train)

    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),
        'subsample':trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree', 0.1, 1),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    history = xgb.cv(param, dtrain, num_boost_round=2000, 
                     early_stopping_rounds=100,
                     callbacks=[pruning_callback],
                     metrics='rmse', 
                     folds=cv)

    mean_score = history["test-rmse-mean"].values[-1]
    return mean_score

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
study = optuna.create_study(pruner=pruner, direction='minimize')
study.optimize(objective, n_trials=1000)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

```python id="NsHKTHFSIm49" outputId="e7f61f7b-c6e3-4db5-c6fa-45bfb3439dae" colab={"base_uri": "https://localhost:8080/"}
# Get best params then add to param_1
study_1_params = study.best_params
param_1 = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.1,
}
param_1.update(study_1_params)
param_1
```

```python id="yWR-ZmE3WxaI" outputId="566fe29a-7309-451f-8d82-1ec863e5299d" colab={"base_uri": "https://localhost:8080/"}
dtrain = xgb.DMatrix(X_train_full, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = xgb.cv(
    param_1, dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=100,
    metrics='rmse',
    folds=cv
)
n_estimators_1 = history.shape[0]
n_estimators_1
```

<!-- #region id="MQ-p-_ykjMF0" -->
#### Study 2
<!-- #endregion -->

```python id="MzSy2WLUNqhN" outputId="d4d8cc07-afbe-4db4-d750-8fcb1e893a8c" colab={"base_uri": "https://localhost:8080/"}
def objective(trial):

    dtrain = xgb.DMatrix(X_train_full, label=y_train)

    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.01,
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),
        'subsample':trial.suggest_uniform('subsample', 0.1, 1),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree', 0.1, 1),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    history = xgb.cv(param, dtrain, num_boost_round=2000, 
                     early_stopping_rounds=100,
                     callbacks=[pruning_callback],
                     metrics='rmse', 
                     folds=cv)

    mean_score = history["test-rmse-mean"].values[-1]
    return mean_score

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
study = optuna.create_study(pruner=pruner, direction='minimize')
study.optimize(objective, n_trials=1000)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

```python id="ggqbIf3UNqYX" outputId="6c92f0c4-fe72-494f-e1d9-2ca7e137c58a" colab={"base_uri": "https://localhost:8080/"}
# Get best params then add to param_2
study_2_params = study.best_params
param_2 = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.01,
}
param_2.update(study_2_params)
param_2
```

```python id="C1CGZcryWz4x" outputId="ca88de46-9c54-4f17-c1bf-1e287e17e69e" colab={"base_uri": "https://localhost:8080/"}
dtrain = xgb.DMatrix(X_train_full, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = xgb.cv(
    param_2, dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=100,
    metrics='rmse',
    folds=cv
)
n_estimators_2 = history.shape[0]
n_estimators_2
```

<!-- #region id="46Vnrg0cjRio" -->
#### Evaluation
<!-- #endregion -->

```python id="OLFchLJDXSSE" outputId="03276147-a466-4fce-a76f-bea368138742" colab={"base_uri": "https://localhost:8080/", "height": 111}
xgb_study_1 = XGBRegressor(**param_1, n_estimators=n_estimators_1)
xgb_study_2 = XGBRegressor(**param_2, n_estimators=n_estimators_2)

models = {
    f'XGBRegressor ({n_estimators_1}) {param_1}': xgb_study_1,
    f'XGBRegressor ({n_estimators_2}) {param_2}': xgb_study_2
}
result = evaluate_model(models, X_train_full, X_test_full, y_train, y_test)
result
```

```python id="FxLJ-WOtcgVH"
result.to_csv("tuning_imputed_all (XGB).csv", index=False)
```

<!-- #region id="Cp5UE4lbMZf1" -->
### LightGBM
<!-- #endregion -->

<!-- #region id="YMGLpL8nMZgJ" -->
#### Study 1
<!-- #endregion -->

```python id="_DBuAfcmMZgK" outputId="c9919557-8e91-4d8b-c1fb-9a0fb468a85d" colab={"base_uri": "https://localhost:8080/"}
def objective(trial):

    dtrain = lgb.Dataset(X_train_full, label=y_train)

    params = {
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = lgb.cv(params, dtrain, 
                    num_boost_round=2000, 
                    early_stopping_rounds=100,
                    callbacks=[pruning_callback],
                    metrics='rmse', 
                    folds=cv)

    mean_score = scores['rmse-mean'][-1]
    return mean_score

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
study = optuna.create_study(pruner=pruner, direction='minimize')
study.optimize(objective, n_trials=1000)


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

```python id="w-RaxMEsMZgS" outputId="64119a36-e56a-4e68-d3bc-1eb494228fee" colab={"base_uri": "https://localhost:8080/"}
# Get best params then add to param_1
study_1_params = study.best_params
param_1 = {
    "objective": "regression",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
}
param_1.update(study_1_params)
param_1
```

```python id="kY2PHvvqMZgW" outputId="aed2d556-8c36-4677-8728-3c5e5f9765ca" colab={"base_uri": "https://localhost:8080/"}
dtrain = lgb.Dataset(X_train_full, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = lgb.cv(
    param_1, dtrain, 
    num_boost_round=2000,
    early_stopping_rounds=100,
    metrics='rmse', 
    folds=cv
)
n_estimators_1 = pd.DataFrame(history).shape[0]
n_estimators_1
```

<!-- #region id="8m8NA4IEMZgZ" -->
#### Study 2
<!-- #endregion -->

```python id="5ofs6sGjMZgb" outputId="a6777166-49ad-4074-8356-26bf13947d75" colab={"base_uri": "https://localhost:8080/"}
def objective(trial):

    dtrain = lgb.Dataset(X_train_full, label=y_train)

    params = {
        "objective": "regression",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.01,
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = lgb.cv(params, dtrain, 
                    num_boost_round=2000, 
                    early_stopping_rounds=100,
                    callbacks=[pruning_callback],
                    metrics='rmse', 
                    folds=cv)

    mean_score = scores['rmse-mean'][-1]
    return mean_score

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
study = optuna.create_study(pruner=pruner, direction='minimize')
study.optimize(objective, n_trials=1000)


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

```python id="PEUnEOY2MZgf" outputId="2b1374b9-2175-4a17-dad4-477fa1ff3c77" colab={"base_uri": "https://localhost:8080/"}
# Get best params then add to param_2
study_2_params = study.best_params
param_2 = {
    "objective": "regression",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
}
param_2.update(study_2_params)
param_2
```

```python id="31_EVr9BMZgi" outputId="20238999-c01e-40f7-c0e7-596b748b73a6" colab={"base_uri": "https://localhost:8080/"}
dtrain = lgb.Dataset(X_train_full, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = lgb.cv(
    param_2, dtrain, 
    num_boost_round=2000,
    early_stopping_rounds=100,
    metrics='rmse', 
    folds=cv
)
n_estimators_2 = pd.DataFrame(history).shape[0]
n_estimators_2
```

<!-- #region id="kdnBKWshMZgl" -->
#### Evaluation
<!-- #endregion -->

```python id="L323nvX2MZgm" outputId="8e1f1d2b-2124-4037-b0be-9c1ebadc5b9d" colab={"base_uri": "https://localhost:8080/", "height": 111}
lgb_study_1 = LGBMRegressor(**param_1, n_estimators=n_estimators_1)
lgb_study_2 = LGBMRegressor(**param_2, n_estimators=n_estimators_2)

models = {
    f'LGBMRegressor ({n_estimators_1}) {param_1}': lgb_study_1,
    f'LGBMRegressor ({n_estimators_2}) {param_2}': lgb_study_2
}
result = evaluate_model(models, X_train, X_test, y_train, y_test)
result
```

<!-- #region id="PMLdn24uMZgq" -->
#### Study 3
<!-- #endregion -->

```python id="1yqpJRkqMZgr" outputId="003e7b8c-0df8-4569-abb2-f6f961e07203" colab={"base_uri": "https://localhost:8080/", "height": 510}
def objective(trial):

    dtrain = lgb.Dataset(X_train_full, label=y_train)

    param = param_2
    param["learning_rate"] = trial.suggest_uniform('learning_rate', 0.001, 0.01)

    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "rmse")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = lgb.cv(params, dtrain, 
                    num_boost_round=2000, 
                    early_stopping_rounds=100,
                    callbacks=[pruning_callback],
                    metrics='rmse', 
                    folds=cv)

    mean_score = scores['rmse-mean'][-1]
    return mean_score

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
study = optuna.create_study(pruner=pruner, direction='minimize')
study.optimize(objective, n_trials=1000)


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

```python id="hRrJ92p_MZgu"
# Get best params then add to param_3
param_3 = param_2.copy()
param_3["learning_rate"] = study.best_params["learning_rate"]
param_3
```

```python id="kVNd4F6mMZgx"
dtrain = lgb.Dataset(X_train_full, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = lgb.cv(
    param_3, dtrain, 
    num_boost_round=2000,
    early_stopping_rounds=100,
    metrics='rmse', 
    folds=cv
)
n_estimators_3 = pd.DataFrame(history).shape[0]
n_estimators_3
```

<!-- #region id="_gJqhWdzMZg0" -->
#### Evaluation
<!-- #endregion -->

```python id="Ita98N6JMZg1"
lgb_study_1 = LGBMRegressor(**param_1, n_estimators=n_estimators_1)
lgb_study_2 = LGBMRegressor(**param_2, n_estimators=n_estimators_2)
lgb_study_3 = LGBMRegressor(**param_3, n_estimators=n_estimators_3)

models = {
    f'LGBMRegressor ({n_estimators_1}) {param_1}': lgb_study_1,
    f'LGBMRegressor ({n_estimators_2}) {param_2}': lgb_study_2,
    f'LGBMRegressor ({n_estimators_3}) {param_3}': lgb_study_3
}
result = evaluate_model(models, X_train_full, X_test_full, y_train, y_test)
result
```

```python id="yALoggpxMZg4"
result.to_csv("tuning_dropna_all (LGB).csv", index=False)
```
