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
#     name: python3
# ---

# %% [markdown] id="IN1jfOnfZIqT"
# # Data Importing

# %% cell_id="00035-adb3a37b-199a-4d0e-ba89-ea8c10843673" execution={"iopub.execute_input": "2020-10-13T13:29:52.833962Z", "iopub.status.busy": "2020-10-13T13:29:52.833962Z", "iopub.status.idle": "2020-10-13T13:29:52.864878Z", "shell.execute_reply": "2020-10-13T13:29:52.863881Z", "shell.execute_reply.started": "2020-10-13T13:29:52.833962Z"} id="bKob_zWgIakl" output_cleared=false tags=[] outputId="abad69f7-6d95-47a7-f39d-aa2c323d8c2a" colab={"base_uri": "https://localhost:8080/"}
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

# %% id="uOHFykGLHlTs" outputId="e29d793b-087b-4151-8549-6804e76f5f13" colab={"base_uri": "https://localhost:8080/", "height": 204}
df = pd.read_csv('../data/processed/after_prep.csv')
df.head()

# %% id="Q7_R-5ZmHlTx" outputId="f21a5ba3-a2d8-4b27-a719-8652785c2256" colab={"base_uri": "https://localhost:8080/"}
df.info()

# %% [markdown] id="g1GS1AAUZIt9"
# # Preprocessing

# %% execution={"iopub.execute_input": "2020-10-13T13:29:52.867870Z", "iopub.status.busy": "2020-10-13T13:29:52.866875Z", "iopub.status.idle": "2020-10-13T13:29:52.879124Z", "shell.execute_reply": "2020-10-13T13:29:52.879124Z", "shell.execute_reply.started": "2020-10-13T13:29:52.867870Z"} id="INV8VvOYZItN" outputId="d23b96f5-b584-4bdf-fef6-0c4fe60ffacf" colab={"base_uri": "https://localhost:8080/"}
# Delete outlier
df = df[~(df.Kilometers_Driven > 1e6)]
df.shape

# %% execution={"iopub.execute_input": "2020-10-13T13:29:52.879124Z", "iopub.status.busy": "2020-10-13T13:29:52.879124Z", "iopub.status.idle": "2020-10-13T13:29:52.910917Z", "shell.execute_reply": "2020-10-13T13:29:52.909951Z", "shell.execute_reply.started": "2020-10-13T13:29:52.879124Z"} id="TYqvFHW1HqFX" outputId="73c8156d-d213-41d0-f2ca-5d8e6c64c1cf" colab={"base_uri": "https://localhost:8080/", "height": 483}
# Drop missing values
df = df.dropna()
null_checker(df)

# %% [markdown] id="Fvvwh2uSWvUU"
# ## Feature enginering

# %% id="bUXLzqVHZOI9"
# Grouping category less than 10 to "Other"
for col in ["Brand", "Series", "Type"]:
    counts = df[col].value_counts()
    other = counts[counts < 10].index
    df[col] = df[col].replace(other, "Other")

# %% id="2HEc8zicU0uy" outputId="eeeba314-bd64-46d3-b9c8-a492605a3937" colab={"base_uri": "https://localhost:8080/", "height": 241}
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
    
df.head()

# %% [markdown] id="yEgVyyNSZIt9"
# ## Train test split

# %% execution={"iopub.execute_input": "2020-10-13T13:29:52.911913Z", "iopub.status.busy": "2020-10-13T13:29:52.911913Z", "iopub.status.idle": "2020-10-13T13:29:52.926873Z", "shell.execute_reply": "2020-10-13T13:29:52.925908Z", "shell.execute_reply.started": "2020-10-13T13:29:52.911913Z"} id="nPxFt6bSZIt-"
# melakukan train test split di awal untuk mencegah data bocor ke test set saat dilakukan encoding/imputation
train_data, test_data = train_test_split(df, test_size=0.25, random_state=0)

# %% [markdown] id="oxqsMHrKZIuA"
# ## Encoding

# %% cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" execution={"iopub.execute_input": "2020-10-13T13:29:52.928873Z", "iopub.status.busy": "2020-10-13T13:29:52.927872Z", "iopub.status.idle": "2020-10-13T13:29:53.107446Z", "shell.execute_reply": "2020-10-13T13:29:53.106483Z", "shell.execute_reply.started": "2020-10-13T13:29:52.928873Z"} id="_0criLnZIakn" output_cleared=false tags=[]
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
train_data["Owner_Type"] = train_data["Owner_Type"].map(mapping_owner)
train_data["Transmission"] = train_data["Transmission"].map(mapping_trans)
# Encoding test set
test_data["Owner_Type"] = test_data["Owner_Type"].map(mapping_owner)
test_data["Transmission"] = test_data["Transmission"].map(mapping_trans)

# %% id="w_QmQ1-COIRX" outputId="75000f20-1597-48fd-b81b-e41310aec223" colab={"base_uri": "https://localhost:8080/"}
import kfold_target_encoder as enc
col_to_encode = train_data.select_dtypes("object").columns.tolist()
col_to_encode

# Encoding train set
for col in col_to_encode:
    targetc = enc.KFoldTargetEncoderTrain(col, "Price", n_fold=5)
    train_data = targetc.fit_transform(train_data)

# Encoding test set
for col in col_to_encode:
    test_targetc = enc.KFoldTargetEncoderTest(train_data, col, col+"_Enc")
    test_data = test_targetc.fit_transform(test_data)

# Delete old features
train_data.drop(columns=col_to_encode, inplace=True)
test_data.drop(columns=col_to_encode, inplace=True)

# %%
test_data.info()


# %% [markdown] id="wV2sjkqEZIup"
# # Modeling

# %% [markdown] id="4g_nWqotKl6_"
# ## Functions

# %% execution={"iopub.execute_input": "2020-10-13T13:29:53.181043Z", "iopub.status.busy": "2020-10-13T13:29:53.181043Z", "iopub.status.idle": "2020-10-13T13:29:53.195221Z", "shell.execute_reply": "2020-10-13T13:29:53.194224Z", "shell.execute_reply.started": "2020-10-13T13:29:53.181043Z"} id="Qp4QHIuFZIuq"
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


# %% execution={"iopub.execute_input": "2020-10-13T13:29:53.198250Z", "iopub.status.busy": "2020-10-13T13:29:53.197220Z", "iopub.status.idle": "2020-10-13T13:29:53.212177Z", "shell.execute_reply": "2020-10-13T13:29:53.210182Z", "shell.execute_reply.started": "2020-10-13T13:29:53.198250Z"} id="BXEr8F5VZIu0"
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


# %%
X_train = train_data.drop(columns="Price")
y_train = train_data["Price"]
X_test = test_data.drop(columns="Price")
y_test = test_data["Price"]

# %%
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# %% [markdown] id="UR-TUfLrHlUy"
# ## Hyperparameter Tuning

# %% [markdown] id="kGwkRRZ8W43D"
# ### XGBoost

# %% [markdown] id="PzcpvsXbjHs3"
# #### Study 1

# %% id="qwWoTc--ImFJ" outputId="906e5e9c-0da6-4023-8c34-cb82002d6584" colab={"base_uri": "https://localhost:8080/"}
def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
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

# %% id="NsHKTHFSIm49" outputId="49057bf4-8362-4344-b6a3-389aa5c49421" colab={"base_uri": "https://localhost:8080/"}
# Get best params then add to param_1
study_1_params = study.best_params
param_1 = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.1,
}
param_1.update(study_1_params)
param_1

# %% id="yWR-ZmE3WxaI" outputId="a4d4dae9-188e-4b13-92fc-e9b2709bade5" colab={"base_uri": "https://localhost:8080/"}
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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


# %% [markdown] id="MQ-p-_ykjMF0"
# #### Study 2

# %% id="MzSy2WLUNqhN" outputId="6fffc654-90d5-48ad-a432-13b63a4aee74" colab={"base_uri": "https://localhost:8080/"}
def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

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

# %% id="ggqbIf3UNqYX" outputId="329a109b-d05c-4499-e932-454e68d31aea" colab={"base_uri": "https://localhost:8080/"}
# Get best params then add to param_2
study_2_params = study.best_params
param_2 = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'learning_rate': 0.01,
}
param_2.update(study_2_params)
param_2

# %% id="C1CGZcryWz4x" outputId="25d50256-fd90-4b29-99c5-043e041573bf" colab={"base_uri": "https://localhost:8080/"}
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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

# %% [markdown] id="46Vnrg0cjRio"
# #### Evaluation

# %% id="OLFchLJDXSSE" outputId="052d7cff-1925-462b-8d1a-b5ec8e103965" colab={"base_uri": "https://localhost:8080/"}
xgb_study_1 = XGBRegressor(**param_1, n_estimators=n_estimators_1)
xgb_study_2 = XGBRegressor(**param_2, n_estimators=n_estimators_2)

models = {
    f'XGBRegressor ({n_estimators_1})': xgb_study_1,
    f'XGBRegressor ({n_estimators_2})': xgb_study_2
}
evaluate_model(models, X_train, X_test, y_train, y_test)


# %% [markdown] id="0SoMHIJlYBFk"
# #### Study 3

# %% id="a2gYuFxmYBF4" outputId="99d7eaf5-c779-40e4-f4ee-d0580294452a" colab={"base_uri": "https://localhost:8080/"}
def objective(trial):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    param = param_2
    param["learning_rate"] = trial.suggest_uniform('learning_rate', 0.001, 0.01)

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

# %% id="Dp9QnijQYBGD"
# Get best params then add to param_3
param_3 = param_2.copy()
param_3["learning_rate"] = study.best_params["learning_rate"]
param_3

# %% id="H03n4RSjYBGH"
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

cv = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=0
)
history = xgb.cv(
    param_3, dtrain, 
    num_boost_round=2000, 
    early_stopping_rounds=100,
    metrics='rmse',
    folds=cv
)
n_estimators_3 = history.shape[0]
n_estimators_3

# %% [markdown] id="ty69r92GYXtx"
# #### Evaluation

# %% id="bQp80jXaYXuG"
xgb_study_1 = XGBRegressor(**param_1, n_estimators=n_estimators_1)
xgb_study_2 = XGBRegressor(**param_2, n_estimators=n_estimators_2)
xgb_study_3 = XGBRegressor(**param_3, n_estimators=n_estimators_3)

models = {
    f'XGBRegressor ({n_estimators_1}) {param_1}': xgb_study_1,
    f'XGBRegressor ({n_estimators_2}) {param_2}': xgb_study_2,
    f'XGBRegressor ({n_estimators_3}) {param_3}': xgb_study_3
}
result = evaluate_model(models, X_train, X_test, y_train, y_test)
result

# %% id="SCciIDdwdDd8"
result.to_csv("tuning_dropna_all (XGB).csv", index=False)


# %% [markdown] id="Cp5UE4lbMZf1"
# ### LightGBM

# %% [markdown] id="YMGLpL8nMZgJ"
# #### Study 1

# %% id="_DBuAfcmMZgK"
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

# %% id="w-RaxMEsMZgS"
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

# %% id="kY2PHvvqMZgW"
dtrain = lgb.Dataset(X_train, label=y_train)

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


# %% [markdown] id="8m8NA4IEMZgZ"
# #### Study 2

# %% id="5ofs6sGjMZgb"
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

# %% id="PEUnEOY2MZgf"
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

# %% id="31_EVr9BMZgi"
dtrain = lgb.Dataset(X_train, label=y_train)

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

# %% [markdown] id="kdnBKWshMZgl"
# #### Evaluation

# %% id="L323nvX2MZgm"
lgb_study_1 = LGBMRegressor(**param_1, n_estimators=n_estimators_1)
lgb_study_2 = LGBMRegressor(**param_2, n_estimators=n_estimators_2)

models = {
    f'LGBMRegressor ({n_estimators_1}) {param_1}': lgb_study_1,
    f'LGBMRegressor ({n_estimators_2}) {param_2}': lgb_study_2
}
result = evaluate_model(models, X_train, X_test, y_train, y_test)
result


# %% [markdown] id="PMLdn24uMZgq"
# #### Study 3

# %% id="1yqpJRkqMZgr"
def objective(trial):

    dtrain = lgb.Dataset(X_train, label=y_train)

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

# %% id="hRrJ92p_MZgu"
# Get best params then add to param_3
param_3 = param_2.copy()
param_3["learning_rate"] = study.best_params["learning_rate"]
param_3

# %% id="kVNd4F6mMZgx"
dtrain = lgb.Dataset(X_train, label=y_train)

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

# %% [markdown] id="_gJqhWdzMZg0"
# #### Evaluation

# %% id="Ita98N6JMZg1"
lgb_study_1 = LGBMRegressor(**param_1, n_estimators=n_estimators_1)
lgb_study_2 = LGBMRegressor(**param_2, n_estimators=n_estimators_2)
lgb_study_3 = LGBMRegressor(**param_3, n_estimators=n_estimators_3)

models = {
    f'LGBMRegressor ({n_estimators_1}) {param_1}': lgb_study_1,
    f'LGBMRegressor ({n_estimators_2}) {param_2}': lgb_study_2,
    f'LGBMRegressor ({n_estimators_3}) {param_3}': lgb_study_3
}
result = evaluate_model(models, X_train, X_test, y_train, y_test)
result

# %% id="yALoggpxMZg4"
result.to_csv("tuning_dropna_all (LGB).csv", index=False)

# %% id="QNQzROBsMZg7"
