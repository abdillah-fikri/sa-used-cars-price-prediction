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

```python cell_id="00000-22bc0b26-1ea9-4900-a08a-fed218efdabd" output_cleared=false source_hash="9c12a302" execution_millis=554 execution_start=1602829891550
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
```

```python
ver1 = pd.read_csv('../data/processed/summary_dropna_all.csv')
ver2 = pd.read_csv('../data/processed/summary_dropna_selected.csv')
ver3 = pd.read_csv('../data/processed/summary_imputed_all.csv')
ver4 = pd.read_csv('../data/processed/summary_imputed_selected.csv')

df_all_ver = pd.concat([ver1, ver2, ver3, ver4], axis=0)
df_all_ver.sort_values(by='Test RMSE', inplace=True)
df_all_ver.head()
```

```python cell_id="00001-c84496a8-764a-4ebf-9d38-ea2d6e9b0438" output_cleared=false source_hash="93fbbcfd" execution_millis=117 execution_start=1602829939970
df_all_ver = pd.read_excel('Model_Summary.xlsx')
```

```python cell_id="00002-fda41ab5-1578-4759-95fd-34bbab934939" output_cleared=false source_hash="a46fdaaf" execution_millis=82 execution_start=1602829944018
df_all_ver['Dataset Version'].unique()
```

```python tags=[] cell_id="00003-7d415f67-fe9a-4bb4-9bee-fa5d53ac3de1" output_cleared=false source_hash="69479740" execution_millis=115 execution_start=1602852201510
grouped = df_all_ver.groupby(['Model'], as_index=False)[['Train RMSE', 'CV RMSE', 'Test RMSE']].mean()
grouped = round(grouped.sort_values(by='Test RMSE'), 4)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=grouped['Model'], y=grouped['Train RMSE'],
    text=grouped['Train RMSE'], textposition='auto',
    name='Train RMSE'
))
fig.add_trace(go.Bar(
    x=grouped['Model'], y=grouped['CV RMSE'],
    text=grouped['CV RMSE'], textposition='auto',
    name='CV RMSE'
))
fig.add_trace(go.Bar(
    x=grouped['Model'], y=grouped['Test RMSE'],
    text=grouped['Test RMSE'], textposition='auto',
    name='Test RMSE'
))
fig.update_layout(
    title='Average RMSE Score',
    xaxis=dict(title='Model'),
    yaxis=dict(title='RMSE Score')
)
fig.show()
```

```python tags=[] cell_id="00003-5f4ebd9b-cf80-4066-b959-a02203215676" output_cleared=false source_hash="1be6e537" execution_millis=303 execution_start=1602851013982
order_all = {'Dataset Version': [
    'dropna + all + unscaled', 'dropna + all + scaled',
    'imputed + all + unscaled', 'imputed + all + scaled',
    'dropna + selected + unscaled', 'dropna + selected + scaled',
    'imputed + selected + unscaled', 'imputed + selected + scaled'
]}

fig = px.bar(
    df_all_ver, x='Model', y='Test RMSE', color='Dataset Version',
    barmode='group', category_orders=order_all, text='Test RMSE'
)

fig.update_layout(title='RMSE Score on Test Set')

fig.show()
```

```python tags=[] cell_id="00004-e6cc76da-a309-4a59-8136-ad80d3a1ba4d" output_cleared=false source_hash="19d3c8" execution_millis=139 execution_start=1602851042958
fig = px.bar(
    df_all_ver, x='Model', y='Fit Time', color='Dataset Version',
    barmode='group', category_orders=order_all, text='Fit Time',
    log_y=True
)

fig.update_layout(title='Training Time on Cross Validation 5 Folds')

fig.show()
```

```python cell_id="00003-0a221708-2f30-45f5-8507-8b3104cce91e" output_cleared=false source_hash="d1533531" execution_millis=211 execution_start=1602850857283
filtered = df_all_ver[df_all_ver['Dataset Version'].str.contains('all')]

order = {'Dataset Version': ['dropna + all + unscaled', 'dropna + all + scaled',
                             'imputed + all + unscaled', 'imputed + all + scaled']}

fig = px.bar(filtered, x='Model', y='Test RMSE', color='Dataset Version',
             barmode='group', category_orders=order, text='Test RMSE')

fig.update_layout(title='Model RMSE Score on Test Set (all features only)')

fig.show()
```

```python cell_id="00004-654fb49d-5f40-4c19-b413-46e0483706b6"

```
