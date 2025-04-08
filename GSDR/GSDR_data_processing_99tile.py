import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

df = pd.read_csv('sumup_GSDR.csv')

df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')
df_season = df[df['datetime'].dt.month.isin([6, 7, 8])]
df_season.iloc[:, 1:] = df_season.iloc[:, 1:].applymap(lambda x: x if x >= 0.1 else np.nan)

quantiles_99 = df_season.iloc[:,1:].quantile(0.99)
df_99tile = df_season.iloc[:,1:].apply(lambda x: x.where(x > quantiles_99[x.name]))
df_99tile.insert(0, 'datetime', df_season.iloc[:, 0])


df_99tile['datetime'] = pd.to_datetime(df_99tile['datetime'])
df_99tile['Year'] = df_99tile['datetime'].dt.year
df_99tile['Hour'] = df_99tile['datetime'].dt.hour
df_99tile = df_99tile.drop(columns=['datetime'])
df_avg_hourly = df_99tile.groupby(['Year', 'Hour'], as_index=False).mean()


df_avg_hourly.to_csv('EP99_GSDR.csv')