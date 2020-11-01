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

```python cell_id="00000-22bc0b26-1ea9-4900-a08a-fed218efdabd" execution_millis=554 execution_start=1602829891550 output_cleared=false source_hash="9c12a302"
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "presentation"
pio.templates
```

```python
ver1 = pd.read_csv('../data/processed/summary_dropna_all.csv')
ver2 = pd.read_csv('../data/processed/summary_dropna_selected.csv')
ver3 = pd.read_csv('../data/processed/summary_imputed_all.csv')
ver4 = pd.read_csv('../data/processed/summary_imputed_selected.csv')

df_all_ver = pd.concat([ver1, ver2, ver3, ver4], axis=0)
df_all_ver.sort_values(by='CV RMSE', inplace=True)
df_all_ver = round(df_all_ver, 3)
df_all_ver = df_all_ver[~((df_all_ver["CV RMSE"] > 100) | (df_all_ver["Test RMSE"] > 100))]
df_all_ver.head(10)
```

```python cell_id="00002-fda41ab5-1578-4759-95fd-34bbab934939" execution_millis=82 execution_start=1602829944018 output_cleared=false source_hash="a46fdaaf"
df_all_ver['Dataset Version'].unique()
```

```python cell_id="00003-7d415f67-fe9a-4bb4-9bee-fa5d53ac3de1" execution_millis=115 execution_start=1602852201510 output_cleared=false source_hash="69479740" tags=[]
grouped = df_all_ver.groupby(['Model'], as_index=False)[['Train RMSE', 'CV RMSE', 'Test RMSE']].mean()
grouped = round(grouped.sort_values(by='Test RMSE'), 3)

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
    title='Average RMSE Score of Base Model (on 8 dataset version each)',
    xaxis=None,
    yaxis=dict(title='RMSE Score'),
    height=500,
    width=1200,
    margin=dict(l=100, r=100, t=100, b=100),
    font_size=17
)
fig.add_shape(type="rect",
    xref="x", yref="y",
    x0=0.55, y0=0,
    x1=1.45, y1=3.8,
    line=dict(
        color="Red",
        width=3,
    ),
)
fig.add_shape(type="rect",
    xref="x", yref="y",
    x0=4.55, y0=0,
    x1=7.45, y1=6.1,
    line=dict(
        color="Purple",
        width=3,
    ),
)
fig.show()
```

```python cell_id="00003-5f4ebd9b-cf80-4066-b959-a02203215676" execution_millis=303 execution_start=1602851013982 output_cleared=false source_hash="1be6e537" tags=[]
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

fig.update_layout(
    title='RMSE Score on Test Set', 
    height=500,
    width=1200
)

fig.show()
```

```python cell_id="00004-e6cc76da-a309-4a59-8136-ad80d3a1ba4d" execution_millis=139 execution_start=1602851042958 output_cleared=false source_hash="19d3c8" tags=[]
fig = px.bar(
    df_all_ver, x='Model', y='Fit Time', color='Dataset Version',
    barmode='group', category_orders=order_all, text='Fit Time',
    log_y=True
)

fig.update_layout(
    title='Training Time on Cross Validation 5 Folds',
    height=500,
    width=1200
)

fig.show()
```

```python cell_id="00003-0a221708-2f30-45f5-8507-8b3104cce91e" execution_millis=211 execution_start=1602850857283 output_cleared=false source_hash="d1533531"
filtered = df_all_ver[df_all_ver['Dataset Version'].str.contains('all')]

order = {'Dataset Version': ['dropna + all + unscaled', 'dropna + all + scaled',
                             'imputed + all + unscaled', 'imputed + all + scaled']}

fig = px.bar(filtered, x='Model', y='Test RMSE', color='Dataset Version',
             barmode='group', category_orders=order, text='Test RMSE')

fig.update_layout(
    title='Model RMSE Score on Test Set (all features only)',
    height=500,
    width=1200
)

fig.show()
```

<!-- #region cell_id="00004-654fb49d-5f40-4c19-b413-46e0483706b6" -->
## Best 5 Models
<!-- #endregion -->

### Before Tuning

```python
df_all_ver[df_all_ver["Dataset Version"].str.contains("unscaled")].head(10)
```

```python
def replace_ver(data):
    if data == "dropna + all + unscaled":
        return "Ver1"
    elif data == "imputed + all + unscaled":
        return "Ver2"
    elif data == "dropna + selected + unscaled":
        return "Ver3"
    else:
        return "Ver4"
```

```python
best_10_before = df_all_ver[df_all_ver["Dataset Version"].str.contains("unscaled")].head(10)
best_10_before["Dataset Version"] = best_10_before["Dataset Version"].apply(replace_ver)
best_10_before["Model"] = best_10_before["Model"] + " (" + best_10_before["Dataset Version"] + ")"
best_10_before.drop(columns=["Dataset Version"], inplace=True)
best_10_before
```

```python
best_before = df_all_ver[df_all_ver["Dataset Version"].str.contains("unscaled")].head(7)
best_before["Dataset Version"] = best_before["Dataset Version"].apply(replace_ver)
best_before["Model"] = best_before["Model"] + " (" + best_before["Dataset Version"] + ")"
best_before.drop(columns=["Dataset Version"], inplace=True)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=best_before['Model'], y=best_before['Train RMSE'],
    text=best_before['Train RMSE'], textposition='auto',
    name='Train RMSE'
))
fig.add_trace(go.Bar(
    x=best_before['Model'], y=best_before['CV RMSE'],
    text=best_before['CV RMSE'], textposition='auto',
    name='CV RMSE'
))
fig.add_trace(go.Bar(
    x=best_before['Model'], y=best_before['Test RMSE'],
    text=best_before['Test RMSE'], textposition='auto',
    name='Test RMSE'
))
fig.update_layout(
    title='Before Tuning',
    xaxis=None,
    yaxis=dict(title='RMSE Score'),
    height=500*1.2,
    width=600*1.2,
    margin=dict(l=100, r=100, t=70, b=120),
    font_size=15,
    showlegend=False
)
fig.show()
```

```python
fig = go.Figure()

fig.add_trace(go.Bar(
    x=best_before['Model'], y=best_before['Train R2'],
    text=best_before['Train R2'], textposition='auto',
    name='Train R2'
))
fig.add_trace(go.Bar(
    x=best_before['Model'], y=best_before['CV R2'],
    text=best_before['CV R2'], textposition='auto',
    name='CV R2'
))
fig.add_trace(go.Bar(
    x=best_before['Model'], y=best_before['Test R2'],
    text=best_before['Test R2'], textposition='auto',
    name='Test R2'
))
fig.update_layout(
    title='Before Tuning',
    xaxis=None,
    yaxis=dict(title='RMSE Score'),
    height=500*1.2,
    width=600*1.2,
    margin=dict(l=100, r=100, t=100, b=120),
    font_size=15,
)
fig.show()
```

### After Tuning

```python
ver1_tuned = pd.read_csv("../data/processed/tuning_dropna_all (XGB+LGB).csv")
ver1_tuned
```

```python
ver1_tuned_top = round(ver1_tuned.iloc[[0,2]], 3)
ver1_tuned_top["Model"] = ver1_tuned_top["Model"].apply(lambda x: x.split()[0]) + "_Tuned (Ver1)"
ver1_tuned_top["Model"] = ver1_tuned_top["Model"].str.replace("LGBMRegressor", "LightGBM")
ver1_tuned_top["Model"] = ver1_tuned_top["Model"].str.replace("XGBRegressor", "XGBoost")
ver1_tuned_top
```

```python
ver2_tuned = pd.read_csv("../data/processed/tuning_imputed_all (XGB+LGB).csv")
ver2_tuned
```

```python
ver2_tuned_top = round(ver2_tuned.iloc[[0,2]], 3)
ver2_tuned_top["Model"] = ver2_tuned_top["Model"].apply(lambda x: x.split()[0]) + "_Tuned (Ver2)"
ver2_tuned_top["Model"] = ver2_tuned_top["Model"].str.replace("LGBMRegressor", "LightGBM")
ver2_tuned_top["Model"] = ver2_tuned_top["Model"].str.replace("XGBRegressor", "XGBoost")
ver2_tuned_top
```

```python
best_after = pd.concat([best_10_before, ver1_tuned_top, ver2_tuned_top], axis=0)
best_after = best_after.sort_values(by="CV RMSE").head(7).reset_index(drop=True)
best_after
```

```python
print("D3:", px.colors.qualitative.D3)
print("Plotly:", px.colors.qualitative.Plotly)
```

```python
colors1 = ['#636EFA','#636EFA','#636EFA','#636EFA','#636EFA','#636EFA','#636EFA']
colors2 = ['#EF553B','#EF553B','#EF553B','#EF553B','#EF553B','#EF553B','#EF553B']
colors3 = ['#00CC96','#00CC96','#00CC96','#00CC96','#00CC96','#00CC96','#00CC96']
```

```python
fig = go.Figure()

fig.add_trace(go.Bar(
    x=best_after['Model'], y=best_after['Train RMSE'],
    text=best_after['Train RMSE'], textposition='auto',
    name='Train RMSE',
    #marker_color=colors1
))
fig.add_trace(go.Bar(
    x=best_after['Model'], y=best_after['CV RMSE'],
    text=best_after['CV RMSE'], textposition='auto',
    name='CV RMSE',
    #marker_color=colors2
))
fig.add_trace(go.Bar(
    x=best_after['Model'], y=best_after['Test RMSE'],
    text=best_after['Test RMSE'], textposition='auto',
    name='Test RMSE',
    #marker_color=colors3
))
fig.update_layout(
    title='After Tuning',
    xaxis=None,
    yaxis=dict(
        title=None,
        range=[0,4]
    ),
    height=500*1.2,
    width=600*1.2,
    margin=dict(l=70, r=100, t=70, b=120),
    font_size=15,  
)
fig.add_shape(type="rect",
    xref="x", yref="y",
    x0=-0.47, y0=0,
    x1=1.47, y1=3.03,
    line=dict(
        color="Red",
        width=3,
    ),
)
fig.add_shape(type="rect",
    xref="x", yref="y",
    x0=2.53, y0=0,
    x1=4.47, y1=3.5,
    line=dict(
        color="Purple",
        width=3,
    ),
)
fig.show()
```

```python
fig = go.Figure()

fig.add_trace(go.Bar(
    x=best_after['Model'], y=best_after['Train R2'],
    text=best_after['Train R2'], textposition='auto',
    name='Train R2'
))
fig.add_trace(go.Bar(
    x=best_after['Model'], y=best_after['CV R2'],
    text=best_after['CV R2'], textposition='auto',
    name='CV R2'
))
fig.add_trace(go.Bar(
    x=best_after['Model'], y=best_after['Test R2'],
    text=best_after['Test R2'], textposition='auto',
    name='Test R2'
))
fig.update_layout(
    title='After Tuning',
    xaxis=None,
    yaxis=None,
    height=500*1.2,
    width=600*1.2,
    margin=dict(l=100, r=100, t=70, b=120),
    font_size=16,  
)
fig.show()
```

```python
final_model = best_after.loc[1]
final_model = pd.DataFrame(final_model).T
```

```python
final_model
```

```python
fig = go.Figure()

fig.add_trace(go.Bar(
    x=final_model['Model'], y=final_model['Train RMSE'],
    text=final_model['Train RMSE'], textposition='auto',
    name='Train RMSE',
    #marker_color=colors1
))
fig.add_trace(go.Bar(
    x=final_model['Model'], y=final_model['CV RMSE'],
    text=final_model['CV RMSE'], textposition='auto',
    name='CV RMSE',
    #marker_color=colors2
))
fig.add_trace(go.Bar(
    x=final_model['Model'], y=final_model['Test RMSE'],
    text=final_model['Test RMSE'], textposition='auto',
    name='Test RMSE',
    #marker_color=colors3
))
fig.update_layout(
    title='RMSE Score',
    xaxis=None,
    yaxis=None,
    height=500*1.2,
    width=300*1.2,
    margin=dict(l=70, r=100, t=70, b=120),
    font_size=15, 
    showlegend=False
)
fig.show()
```

```python
fig = go.Figure()

fig.add_trace(go.Bar(
    x=final_model['Model'], y=final_model['Train R2'],
    text=final_model['Train R2'], textposition='auto',
    name='Train R2',
    #marker_color=colors1
))
fig.add_trace(go.Bar(
    x=final_model['Model'], y=final_model['CV R2'],
    text=final_model['CV R2'], textposition='auto',
    name='CV R2',
    #marker_color=colors2
))
fig.add_trace(go.Bar(
    x=final_model['Model'], y=final_model['Test R2'],
    text=final_model['Test R2'], textposition='auto',
    name='Test R2',
    #marker_color=colors3
))
fig.update_layout(
    title='R-Squared Score',
    xaxis=None,
    yaxis=None,
    height=500*1.2,
    width=300*1.2,
    margin=dict(l=70, r=100, t=70, b=120),
    font_size=15, 
    showlegend=False
)
fig.show()
```
