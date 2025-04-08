#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

### lvl 2 setups (systerm)

import os
import numpy as np
import pandas as pd
import xarray as xr


# In[2]:


def circular_mean(hours):
    angles = np.array(hours) * (360 / 24)

    sin_angles = np.sin(np.radians(angles))
    cos_angles = np.cos(np.radians(angles))

    mean_sin = np.mean(sin_angles)
    mean_cos = np.mean(cos_angles)

    mean_angle = np.arctan2(mean_sin, mean_cos) * (180 / np.pi)

    if mean_angle < 0:
        mean_angle += 360

    mean_hour = mean_angle * 24 / 360

    return mean_hour


# In[3]:


df = pd.read_csv('sumup_GSDR.csv')


# In[4]:


df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')
df_season = df[df['datetime'].dt.month.isin([6, 7, 8])]
df_season.iloc[:, 1:] = df_season.iloc[:, 1:].applymap(lambda x: x if x >= 0.1 else np.nan)


# calculate 99tile ---------------
quantiles_99 = df_season.iloc[:,1:].quantile(0.99)
df_99tile = df_season.iloc[:,1:].apply(lambda x: x.where(x > quantiles_99[x.name]))
df_99tile.insert(0, 'datetime', df_season.iloc[:, 0])


# In[5]:


def calculate_circular_means(df_season):

    years = range(1900, 2014)
    stations = df_season.columns[1:]  
    result_df = pd.DataFrame(index=years, columns=stations)
    

    df_season['datetime'] = pd.to_datetime(df_season['datetime'])
    df_season['year'] = df_season['datetime'].dt.year
    df_season['hour'] = df_season['datetime'].dt.hour
    
    for station in stations:
        for year in years:

            yearly_data = df_season[df_season['year'] == year]
            

            rain_hours = yearly_data[station].dropna().index.tolist()
            if rain_hours:
                arr_rain_time = yearly_data.loc[rain_hours, 'hour'].tolist()

                mean_hour = circular_mean(arr_rain_time)
                result_df.at[year, station] = mean_hour

    return result_df
df_circular_mean = calculate_circular_means(df_99tile)
df_circular_mean.to_csv('circular_mean_99tile_GSDR.csv', index=False)


# In[ ]:
quantiles_99_df = pd.DataFrame([quantiles_99], columns=df_season.columns[1:])
quantiles_99_df.to_csv('quantiles_99_threshold_GSDR.csv', index=False)



