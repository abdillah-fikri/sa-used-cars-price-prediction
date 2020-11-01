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

```python cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" colab={"base_uri": "https://localhost:8080/"} execution={"iopub.execute_input": "2020-10-13T13:29:52.833962Z", "iopub.status.busy": "2020-10-13T13:29:52.833962Z", "iopub.status.idle": "2020-10-13T13:29:52.864878Z", "shell.execute_reply": "2020-10-13T13:29:52.863881Z", "shell.execute_reply.started": "2020-10-13T13:29:52.833962Z"} id="bKob_zWgIakl" outputId="cab68c39-a4f2-4aaf-8ecc-d438b5cfbf9b" output_cleared=false tags=[]
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

```python colab={"base_uri": "https://localhost:8080/", "height": 204} id="uOHFykGLHlTs" outputId="673b902e-d3d9-4956-ff8c-37a9a1c072c3"
df = pd.read_csv('../data/processed/after_prep.csv')
df.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q7_R-5ZmHlTx" outputId="6c7d649a-4703-4977-d31b-6d3be82db63c"
df.info()
```

<!-- #region id="g1GS1AAUZIt9" -->
# Preprocessing
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} execution={"iopub.execute_input": "2020-10-13T13:29:52.867870Z", "iopub.status.busy": "2020-10-13T13:29:52.866875Z", "iopub.status.idle": "2020-10-13T13:29:52.879124Z", "shell.execute_reply": "2020-10-13T13:29:52.879124Z", "shell.execute_reply.started": "2020-10-13T13:29:52.867870Z"} id="INV8VvOYZItN" outputId="3cb119d1-6004-45d6-dc0c-ffa16220679e"
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape
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

```python cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" colab={"base_uri": "https://localhost:8080/"} execution={"iopub.execute_input": "2020-10-13T13:29:52.928873Z", "iopub.status.busy": "2020-10-13T13:29:52.927872Z", "iopub.status.idle": "2020-10-13T13:29:53.107446Z", "shell.execute_reply": "2020-10-13T13:29:53.106483Z", "shell.execute_reply.started": "2020-10-13T13:29:52.928873Z"} id="_0criLnZIakn" outputId="664fb5de-971a-4f5c-e353-cf0802ed6e1c" output_cleared=false tags=[]
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

```python colab={"base_uri": "https://localhost:8080/"} id="w_QmQ1-COIRX" outputId="01369227-5f56-4712-f8e9-334e9808a0f1"
# Target Encoding
col_to_encode = ['Series', 'Type']
encoder = ce.TargetEncoder(cols=col_to_encode)
encoder.fit(X_train, y_train)

# Encoding train set
X_train = encoder.transform(X_train)
# Encoding test set
X_test = encoder.transform(X_test)
```

<!-- #region id="YkKs-Cu7dJrL" -->
## Missing Value Imputation
<!-- #endregion -->

```python id="oLY62nxOdNPq"
# memprediksi nilai missing value dengan MICE
imputer = mf.KernelDataSet(
  X_train,
  save_all_iterations=True,
  random_state=1991,
  mean_match_candidates=5
)
imputer.mice(10)
```

```python id="xCuUb1EcdNB2"
# Train set imputation
X_train = imputer.complete_data()
```

```python id="cmggM3hIdMrZ"
# Test set imputation
new_data = imputer.impute_new_data(X_test)
X_test = new_data.complete_data()
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

```python colab={"base_uri": "https://localhost:8080/"} id="qwWoTc--ImFJ" outputId="21c9bb49-0ae2-4d2e-98ec-4223db73e607"
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

```python colab={"base_uri": "https://localhost:8080/"} id="NsHKTHFSIm49" outputId="713fa8e7-5178-4374-8f54-34324ac8c379"
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

```python colab={"base_uri": "https://localhost:8080/"} id="yWR-ZmE3WxaI" outputId="cab31409-7b46-499e-9791-23fd2d47b12d"
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

```python colab={"base_uri": "https://localhost:8080/"} id="MzSy2WLUNqhN" outputId="42a4a3c2-66d5-4d43-8a91-1d56e880f1a4"
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

```python colab={"base_uri": "https://localhost:8080/"} id="ggqbIf3UNqYX" outputId="9a0e34d7-0f5d-484a-ab01-708626c3ab5e"
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

```python colab={"base_uri": "https://localhost:8080/"} id="C1CGZcryWz4x" outputId="0f677d82-5093-4743-8e96-29a54b4d6b72"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 227} id="z3T7iSxoaUPA" outputId="829b8b9d-e0e5-40ff-a6e1-6dfb9e214deb"
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

```python colab={"base_uri": "https://localhost:8080/"} id="a2gYuFxmYBF4" outputId="46f3d28a-393b-4680-df64-8418c6fe05e0"
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

```python colab={"base_uri": "https://localhost:8080/"} id="Dp9QnijQYBGD" outputId="368e32f8-c8b3-4a41-89bf-570de113fcd2"
# Get best params then add to param_3
xgb_param_3 = xgb_param_2.copy()
xgb_param_3["learning_rate"] = study.best_params["learning_rate"]
xgb_param_3
```

```python colab={"base_uri": "https://localhost:8080/"} id="H03n4RSjYBGH" outputId="d895284b-b434-4358-f27b-d970b88f3243"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="bQp80jXaYXuG" outputId="67a816fd-909c-4e24-e931-3c0ea696fd46"
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
xgb_result.to_csv("../data/processed/tuning_imputed_all (XGB).csv", index=False)
```

<!-- #region id="Cp5UE4lbMZf1" -->
### LightGBM
<!-- #endregion -->

<!-- #region id="YMGLpL8nMZgJ" -->
#### Study 1
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="_DBuAfcmMZgK" outputId="981ce043-a99c-4e29-9295-9c13a728de0a"
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

```python colab={"base_uri": "https://localhost:8080/"} id="w-RaxMEsMZgS" outputId="f869abbb-0939-4121-cc46-f7e62aa47881"
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

```python colab={"base_uri": "https://localhost:8080/"} id="kY2PHvvqMZgW" outputId="1d02486f-cc45-4228-e427-6aa904d16e13"
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

```python colab={"base_uri": "https://localhost:8080/"} id="5ofs6sGjMZgb" outputId="b5de9e48-8a13-4150-ee0f-4265b20f3433"
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

```python colab={"base_uri": "https://localhost:8080/"} id="PEUnEOY2MZgf" outputId="c2e4860e-72a8-4c63-a0bf-4a90d4df7370"
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

```python colab={"base_uri": "https://localhost:8080/"} id="8i-_ufqKIX8O" outputId="84532adb-9e66-4954-920b-465807986d37"
lgb_param_2['learning_rate'] = 0.01
lgb_param_2
```

```python colab={"base_uri": "https://localhost:8080/"} id="31_EVr9BMZgi" outputId="df603c1b-0b7a-4a5d-a00e-012254fd3094"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 227} id="L323nvX2MZgm" outputId="0a736a77-f7fd-48cb-cc28-4a6c6b60c7f8"
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

```python colab={"base_uri": "https://localhost:8080/"} id="1yqpJRkqMZgr" outputId="f5484a17-095b-4d46-b369-865b1953c862"
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

```python colab={"base_uri": "https://localhost:8080/"} id="hRrJ92p_MZgu" outputId="d6bad2e0-fcb1-4fba-9145-a604bdb5cb87"
# Get best params then add to param_3
lgb_param_3 = lgb_param_2.copy()
lgb_param_3["learning_rate"] = study.best_params["learning_rate"]
lgb_param_3
```

```python colab={"base_uri": "https://localhost:8080/"} id="kVNd4F6mMZgx" outputId="3aa1f765-3300-4688-b06c-b256823fa149"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="Ita98N6JMZg1" outputId="7dd0be9e-f414-4450-f6ae-36152c2a2cb8"
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
lgb_result.to_csv("tuning_imputed_all (LGB).csv", index=False)
```

<!-- #region id="ke3wHorVbjZ4" -->
## Combine Result
<!-- #endregion -->

```python id="EGQKeyKQZilg"
combined_result = pd.concat([xgb_result, lgb_result], axis=0)
combined_result.sort_values(by='CV RMSE', inplace=True)
combined_result.to_csv("../data/processed/tuning_imputed_all (XGB+LGB).csv", index=True)
```

```python id="Mo853cR-cIPU"

```
