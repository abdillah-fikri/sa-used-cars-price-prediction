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
    name: python3
---

<!-- #region id="IN1jfOnfZIqT" -->
# Data Importing
<!-- #endregion -->

```python cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" execution={"iopub.execute_input": "2020-10-13T13:29:52.833962Z", "iopub.status.busy": "2020-10-13T13:29:52.833962Z", "iopub.status.idle": "2020-10-13T13:29:52.864878Z", "shell.execute_reply": "2020-10-13T13:29:52.863881Z", "shell.execute_reply.started": "2020-10-13T13:29:52.833962Z"} id="bKob_zWgIakl" output_cleared=false tags=[]
import numpy as np
import pandas as pd
import category_encoders as ce
import miceforest as mf
import optuna
import lightgbm as lgb
import xgboost as xgb

from utils import null_checker, evaluate_model
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="uOHFykGLHlTs" outputId="800ad260-330c-44bc-8f2e-3b4bbf437b80"
df = pd.read_csv('after_prep.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q7_R-5ZmHlTx" outputId="c8adec21-7910-4979-fb09-f297e2612267"
df.info()
```

<!-- #region id="g1GS1AAUZIt9" -->
# Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} execution={"iopub.execute_input": "2020-10-13T13:29:52.867870Z", "iopub.status.busy": "2020-10-13T13:29:52.866875Z", "iopub.status.idle": "2020-10-13T13:29:52.879124Z", "shell.execute_reply": "2020-10-13T13:29:52.879124Z", "shell.execute_reply.started": "2020-10-13T13:29:52.867870Z"} id="INV8VvOYZItN" outputId="5c9e5a34-8aa4-48e8-d66e-869ee5db7232"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 483} execution={"iopub.execute_input": "2020-10-13T13:29:52.879124Z", "iopub.status.busy": "2020-10-13T13:29:52.879124Z", "iopub.status.idle": "2020-10-13T13:29:52.910917Z", "shell.execute_reply": "2020-10-13T13:29:52.909951Z", "shell.execute_reply.started": "2020-10-13T13:29:52.879124Z"} id="TYqvFHW1HqFX" outputId="1c62a494-28b3-4785-d101-82ab26ddf4dc"
# Drop missing values
df = df.dropna()
null_checker(df)
```

<!-- #region id="Fvvwh2uSWvUU" -->
## Feature enginering
<!-- #endregion -->

```python id="bUXLzqVHZOI9"
# Grouping category less than 10 to "Other"
for col in ["Brand", "Series", "Type"]:
    counts = df[col].value_counts()
    other = counts[counts < 10].index
    df[col] = df[col].replace(other, "Other")
```

```python id="2HEc8zicU0uy"
# Make categorical feature interactions
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand']

for col in combinations(cat_cols, 2):
    new_col = col[0]+'_'+col[1]
    df[new_col] = df[col[0]] + "_" + df[col[1]]
    
    counts = df[new_col].value_counts()
    other = counts[counts < 10].index
    df[new_col] = df[new_col].replace(other, "Other")

    encoder = LabelEncoder()
    df[new_col] = encoder.fit_transform(df[new_col])
```

<!-- #region id="yEgVyyNSZIt9" -->
## Train test split
<!-- #endregion -->

```python execution={"iopub.execute_input": "2020-10-13T13:29:52.911913Z", "iopub.status.busy": "2020-10-13T13:29:52.911913Z", "iopub.status.idle": "2020-10-13T13:29:52.926873Z", "shell.execute_reply": "2020-10-13T13:29:52.925908Z", "shell.execute_reply.started": "2020-10-13T13:29:52.911913Z"} id="nPxFt6bSZIt-"
# melakukan train test split di awal untuk mencegah data leakage
features = df.drop(columns=['Price'])
target = df['Price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
```

<!-- #region id="oxqsMHrKZIuA" -->
## Encoding
<!-- #endregion -->

```python cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" colab={"base_uri": "https://localhost:8080/"} execution={"iopub.execute_input": "2020-10-13T13:29:52.928873Z", "iopub.status.busy": "2020-10-13T13:29:52.927872Z", "iopub.status.idle": "2020-10-13T13:29:53.107446Z", "shell.execute_reply": "2020-10-13T13:29:53.106483Z", "shell.execute_reply.started": "2020-10-13T13:29:52.928873Z"} id="_0criLnZIakn" outputId="d8bac6fc-e117-4614-82ff-06b630107b50" output_cleared=false tags=[]
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

```python colab={"base_uri": "https://localhost:8080/"} id="w_QmQ1-COIRX" outputId="0ee7122f-0a35-40c6-f090-ff3f3299f93d"
# Target Encoding
col_to_encode = ['Series', 'Type']
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

<!-- #region id="UR-TUfLrHlUy" -->
## Hyperparameter Tuning
<!-- #endregion -->

<!-- #region id="kGwkRRZ8W43D" -->
### XGBoost
<!-- #endregion -->

<!-- #region id="PzcpvsXbjHs3" -->
#### Study 1
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="qwWoTc--ImFJ" outputId="0168a8dd-bae2-4667-cd01-bb7240c2de29"
def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
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
    history = xgb.cv(params, dtrain, num_boost_round=2000, 
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

```python colab={"base_uri": "https://localhost:8080/"} id="NsHKTHFSIm49" outputId="c7d81757-2cd5-4003-cc37-b646888062dc"
# Get best params then add to param_1
xgb_study_1_params = study.best_params
xgb_param_1 = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.1,
}
xgb_param_1.update(xgb_study_1_params)
xgb_param_1
```

```python colab={"base_uri": "https://localhost:8080/"} id="yWR-ZmE3WxaI" outputId="b4414985-f66e-4981-9794-178d428e76db"
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = xgb.cv(
    xgb_param_1, dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=100,
    metrics='rmse',
    folds=cv
)
xgb_n_estimators_1 = history.shape[0]
xgb_n_estimators_1
```

<!-- #region id="MQ-p-_ykjMF0" -->
#### Study 2
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="MzSy2WLUNqhN" outputId="57cda43d-1fe8-4993-d026-4fbe58a362d3"
def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
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
    history = xgb.cv(params, dtrain, num_boost_round=2000, 
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

```python colab={"base_uri": "https://localhost:8080/"} id="ggqbIf3UNqYX" outputId="6b78f3c2-cc06-4a49-f692-5dd7293349d7"
# Get best params then add to param_2
xgb_study_2_params = study.best_params
xgb_param_2 = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.01,
}
xgb_param_2.update(xgb_study_2_params)
xgb_param_2
```

```python colab={"base_uri": "https://localhost:8080/"} id="C1CGZcryWz4x" outputId="cf5d1998-4da7-47b7-b6f2-5ea9e9cdb1cf"
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = xgb.cv(
    xgb_param_2, dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=100,
    metrics='rmse',
    folds=cv
)
xgb_n_estimators_2 = history.shape[0]
xgb_n_estimators_2
```

<!-- #region id="46Vnrg0cjRio" -->
#### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 227} id="z3T7iSxoaUPA" outputId="79292a28-61a3-4dea-dc41-3d3669808a63"
xgb_study_1 = XGBRegressor(**xgb_param_1, n_estimators=xgb_n_estimators_1)
xgb_study_2 = XGBRegressor(**xgb_param_2, n_estimators=xgb_n_estimators_2)

xgb_models = {
    f'XGBRegressor ({xgb_n_estimators_1}) {xgb_param_1}': xgb_study_1,
    f'XGBRegressor ({xgb_n_estimators_2}) {xgb_param_2}': xgb_study_2
}
evaluate_model(xgb_models, X_train, X_test, y_train, y_test)
```

<!-- #region id="0SoMHIJlYBFk" -->
#### Study 3
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a2gYuFxmYBF4" outputId="e310ec21-be00-4594-a9f0-9c51ba82d344"
def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = xgb_param_2.copy()
    params["learning_rate"] = trial.suggest_uniform('learning_rate', 0.001, 0.01)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    history = xgb.cv(params, dtrain, num_boost_round=2000, 
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

```python colab={"base_uri": "https://localhost:8080/"} id="Dp9QnijQYBGD" outputId="e553402e-6912-4c46-a080-b410424cb618"
# Get best params then add to param_3
xgb_param_3 = xgb_param_2.copy()
xgb_param_3["learning_rate"] = study.best_params["learning_rate"]
xgb_param_3
```

```python colab={"base_uri": "https://localhost:8080/"} id="H03n4RSjYBGH" outputId="284e21be-b77b-487d-a75e-bf49cc5062ac"
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = xgb.cv(
    xgb_param_3, dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=100,
    metrics='rmse',
    folds=cv
)
xgb_n_estimators_3 = history.shape[0]
xgb_n_estimators_3
```

<!-- #region id="ty69r92GYXtx" -->
#### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="bQp80jXaYXuG" outputId="dd042fe9-a67c-431e-8425-2e4398570efe"
xgb_study_1 = XGBRegressor(**xgb_param_1, n_estimators=xgb_n_estimators_1)
xgb_study_2 = XGBRegressor(**xgb_param_2, n_estimators=xgb_n_estimators_2)
xgb_study_3 = XGBRegressor(**xgb_param_3, n_estimators=xgb_n_estimators_3)

xgb_models = {
    f'XGBRegressor ({xgb_n_estimators_1}) {xgb_param_1}': xgb_study_1,
    f'XGBRegressor ({xgb_n_estimators_2}) {xgb_param_2}': xgb_study_2,
    f'XGBRegressor ({xgb_n_estimators_3}) {xgb_param_3}': xgb_study_3
}
xgb_result = evaluate_model(xgb_models, X_train, X_test, y_train, y_test)
xgb_result
```

```python id="SCciIDdwdDd8"
xgb_result.to_csv("tuning_dropna_all (XGB).csv", index=False)
```

<!-- #region id="Cp5UE4lbMZf1" -->
### LightGBM
<!-- #endregion -->

<!-- #region id="YMGLpL8nMZgJ" -->
#### Study 1
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_DBuAfcmMZgK" outputId="a9fd70cf-3394-42a1-9c6c-0fda3bc1dd74"
def objective(trial):

    dtrain = lgb.Dataset(X_train, label=y_train)

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

```python colab={"base_uri": "https://localhost:8080/"} id="w-RaxMEsMZgS" outputId="134d57dd-f374-4ce0-e447-3f00bf782fea"
# Get best params then add to param_1
lgb_study_1_params = study.best_params
lgb_param_1 = {
    "objective": "regression",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
}
lgb_param_1.update(lgb_study_1_params)
lgb_param_1
```

```python colab={"base_uri": "https://localhost:8080/"} id="kY2PHvvqMZgW" outputId="84b6e34a-520a-4bc9-86d3-b7b62c4006e5"
dtrain = lgb.Dataset(X_train, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = lgb.cv(
    lgb_param_1, dtrain, 
    num_boost_round=2000,
    early_stopping_rounds=100,
    metrics='rmse', 
    folds=cv
)
lgb_n_estimators_1 = pd.DataFrame(history).shape[0]
lgb_n_estimators_1
```

<!-- #region id="8m8NA4IEMZgZ" -->
#### Study 2
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5ofs6sGjMZgb" outputId="7998c696-16dc-49a4-a9ad-27f5a9367112"
def objective(trial):

    dtrain = lgb.Dataset(X_train, label=y_train)

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

```python colab={"base_uri": "https://localhost:8080/"} id="PEUnEOY2MZgf" outputId="ebf8ba52-a67e-465e-af85-2061aa5c88d4"
# Get best params then add to param_2
lgb_study_2_params = study.best_params
lgb_param_2 = {
    "objective": "regression",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
}
lgb_param_2.update(lgb_study_2_params)
lgb_param_2
```

```python colab={"base_uri": "https://localhost:8080/"} id="8i-_ufqKIX8O" outputId="45fdfe66-4ef8-4e6d-b5cc-8a7b10f26403"
lgb_param_2['learning_rate'] = 0.01
lgb_param_2
```

```python colab={"base_uri": "https://localhost:8080/"} id="31_EVr9BMZgi" outputId="5128c77f-5ed6-44de-dbc1-08bec21671f2"
dtrain = lgb.Dataset(X_train, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = lgb.cv(
    lgb_param_2, dtrain, 
    num_boost_round=2000,
    early_stopping_rounds=100,
    metrics='rmse', 
    folds=cv
)
lgb_n_estimators_2 = pd.DataFrame(history).shape[0]
lgb_n_estimators_2
```

<!-- #region id="kdnBKWshMZgl" -->
#### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 227} id="L323nvX2MZgm" outputId="06ddabf3-0204-49e5-c4b7-4fa67204d049"
lgb_study_1 = LGBMRegressor(**lgb_param_1, n_estimators=lgb_n_estimators_1)
lgb_study_2 = LGBMRegressor(**lgb_param_2, n_estimators=lgb_n_estimators_2)

lgb_models = {
    f'LGBMRegressor ({lgb_n_estimators_1}) {lgb_param_1}': lgb_study_1,
    f'LGBMRegressor ({lgb_n_estimators_2}) {lgb_param_2}': lgb_study_2
}
evaluate_model(lgb_models, X_train, X_test, y_train, y_test)
```

<!-- #region id="PMLdn24uMZgq" -->
#### Study 3
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1yqpJRkqMZgr" outputId="56ae2288-5844-4198-c39f-74f6f35c4db3"
def objective(trial):

    dtrain = lgb.Dataset(X_train, label=y_train)

    params = lgb_param_2.copy()
    params["learning_rate"] = trial.suggest_uniform('learning_rate', 0.001, 0.01)

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

```python colab={"base_uri": "https://localhost:8080/"} id="hRrJ92p_MZgu" outputId="a4c9d402-3207-455e-b97a-a3cc144c127d"
# Get best params then add to param_3
lgb_param_3 = lgb_param_2.copy()
lgb_param_3["learning_rate"] = study.best_params["learning_rate"]
lgb_param_3
```

```python colab={"base_uri": "https://localhost:8080/"} id="kVNd4F6mMZgx" outputId="ef155009-c4a9-4ba6-84e6-a0e5dea1a0c7"
dtrain = lgb.Dataset(X_train, label=y_train)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = lgb.cv(
    lgb_param_3, dtrain, 
    num_boost_round=2000,
    early_stopping_rounds=100,
    metrics='rmse', 
    folds=cv
)
lgb_n_estimators_3 = pd.DataFrame(history).shape[0]
lgb_n_estimators_3
```

<!-- #region id="_gJqhWdzMZg0" -->
#### Evaluation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="Ita98N6JMZg1" outputId="375f702a-84bf-4add-8545-69126b4dd3ce"
lgb_study_1 = LGBMRegressor(**lgb_param_1, n_estimators=lgb_n_estimators_1)
lgb_study_2 = LGBMRegressor(**lgb_param_2, n_estimators=lgb_n_estimators_2)
lgb_study_3 = LGBMRegressor(**lgb_param_3, n_estimators=lgb_n_estimators_3)

lgb_models = {
    f'LGBMRegressor ({lgb_n_estimators_1}) {lgb_param_1}': lgb_study_1,
    f'LGBMRegressor ({lgb_n_estimators_2}) {lgb_param_2}': lgb_study_2,
    f'LGBMRegressor ({lgb_n_estimators_3}) {lgb_param_3}': lgb_study_3
}
lgb_result = evaluate_model(lgb_models, X_train, X_test, y_train, y_test)
lgb_result
```

```python id="yALoggpxMZg4"
lgb_result.to_csv("tuning_dropna_all (LGB).csv", index=False)
```

<!-- #region id="ke3wHorVbjZ4" -->
## Combine Result
<!-- #endregion -->

```python id="EGQKeyKQZilg"
combined_result = pd.concat([xgb_result, lgb_result], axis=0)
combined_result.sort_values(by='CV RMSE', inplace=True)
combined_result.to_csv("tuning_dropna_all (XGB+LGB).csv", index=True)
```

```python id="Mo853cR-cIPU"

```
