import sys

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

### lvl 2 setups (systerm)

import os
import numpy as np
import pandas as pd
import xarray as xr


df = pd.read_csv('sumup_GSDR.csv')
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H')
df_season = df[df['datetime'].dt.month.isin([6, 7, 8])]
df_season.iloc[:, 1:] = df_season.iloc[:, 1:].applymap(lambda x: x if x >= 0.1 else np.nan)

quantiles_99 = df_season.iloc[:,1:].quantile(0.99)
df_99tile = df_season.iloc[:,1:].apply(lambda x: x.where(x > quantiles_99[x.name]))
df_99tile.insert(0, 'datetime', df_season.iloc[:, 0])


def calculate_yearly_hourly_rainfall_frequency(df_season):
    stations = df_season.columns[1:]  # Exclude the 'datetime' column
    yearly_hourly_rainfall = pd.DataFrame({'year': np.repeat(range(1900, 2014), 24), 'hour': list(range(24)) * 114})
    
    for station in stations:
        station_data = df_season[['datetime', station]].dropna(subset=[station])
        station_data['Year'] = pd.to_datetime(station_data['datetime']).dt.year
        station_data['Hour'] = pd.to_datetime(station_data['datetime']).dt.hour
        
        # Count rainfall occurrences by year and hour
        rainfall_count = station_data.groupby(['Year', 'Hour'])[station].count().reset_index()
        rainfall_pivot = rainfall_count.pivot(index='Year', columns='Hour', values=station).fillna(0)
        
        # Flatten the pivoted data and merge it into the result DataFrame
        rainfall_pivot_flattened = rainfall_pivot.unstack().reset_index(name=station)
        yearly_hourly_rainfall = yearly_hourly_rainfall.merge(rainfall_pivot_flattened, how='left', 
                                                             left_on=['year', 'hour'], right_on=['Year', 'Hour'])
        yearly_hourly_rainfall[station] = yearly_hourly_rainfall[station].fillna(0).astype(int)
        yearly_hourly_rainfall.drop(columns=['Year', 'Hour'], inplace=True)

    return yearly_hourly_rainfall
rainfall_frequency = calculate_yearly_hourly_rainfall_frequency(df_99tile)
rainfall_frequency.to_csv('prec99_hour_count_GSDR.csv')