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
# melakukan train test split di awal untuk mencegah data leakage
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %% [markdown] id="oxqsMHrKZIuA"
# ## Encoding

# %% cell_id="00036-c7e04c20-9ab9-48dc-a699-9e7a06582a8c" execution={"iopub.execute_input": "2020-10-15T12:50:17.900538Z", "iopub.status.busy": "2020-10-15T12:50:17.899538Z", "iopub.status.idle": "2020-10-15T12:50:18.100999Z", "shell.execute_reply": "2020-10-15T12:50:18.100001Z", "shell.execute_reply.started": "2020-10-15T12:50:17.900538Z"} id="_0criLnZIakn" output_cleared=false tags=[]
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

# %%
# One hot encoding for low cardinality feature + Brand
col_to_encode = ['Location', 'Fuel_Type', 'Brand']
oh_encoder = ce.OneHotEncoder(cols=col_to_encode,
                              use_cat_names=True)
oh_encoder.fit(X_train)

# Encoding train set
X_train = oh_encoder.transform(X_train)
# Encoding test set
X_test = oh_encoder.transform(X_test)

# %% execution={"iopub.execute_input": "2020-10-15T12:50:18.102994Z", "iopub.status.busy": "2020-10-15T12:50:18.101997Z", "iopub.status.idle": "2020-10-15T12:50:18.179789Z", "shell.execute_reply": "2020-10-15T12:50:18.178825Z", "shell.execute_reply.started": "2020-10-15T12:50:18.102994Z"} id="kcMLnvJxZIuD"
# Target encoding for high cardinality feature
col_to_encode = X_train.select_dtypes("object").columns
encoder = ce.TargetEncoder(cols=col_to_encode)
encoder.fit(X_train, y_train)

# Encoding train set
X_train = encoder.transform(X_train)
# Encoding test set
X_test = encoder.transform(X_test)

# %% [markdown] id="6MJs1hK7Iv1N"
# ## Missing Value Imputation

# %% execution={"iopub.execute_input": "2020-10-15T12:50:18.181784Z", "iopub.status.busy": "2020-10-15T12:50:18.180785Z", "iopub.status.idle": "2020-10-15T12:50:29.221721Z", "shell.execute_reply": "2020-10-15T12:50:29.221721Z", "shell.execute_reply.started": "2020-10-15T12:50:18.181784Z"} id="ccgkETh_Iv1O"
# memprediksi nilai missing value dengan MICE
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

# %% [markdown] id="aR4Sp3UCZIu2"
# ## Base Model

# %% execution={"iopub.execute_input": "2020-10-15T12:50:31.751211Z", "iopub.status.busy": "2020-10-15T12:50:31.751211Z", "iopub.status.idle": "2020-10-15T12:50:31.765176Z", "shell.execute_reply": "2020-10-15T12:50:31.764178Z", "shell.execute_reply.started": "2020-10-15T12:50:31.751211Z"} id="Oux2OxeDZIu2"
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
