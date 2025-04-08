
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

### lvl 2 setups (systerm)

import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
warnings.filterwarnings('ignore')
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Wedge, Circle
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import datetime
import glob
### T CC

input_folder_t = '/N/project/Zli_lab/gongg/CONUS404_data/LST/JJA/'
base_path = '/N/project/Zli_lab/gongg/CONUS404_data/LST/UTC/'
file_pattern_t = 'T2.wrf2d_d01_????-??-??.nc'
file_pattern_dt = 'TD2.wrf2d_d01_????-??-??.nc'
output = '/N/project/Zli_lab/gongg/CONUS404_data/LST/JJA_dailydata/'


folder_names = ['U-68', 'U-78', 'U-88', 
     # 'U-50', 'U-51', 'U-52', 'U-53', 'U-54', 'U-55', 'U-56', 'U-57', 'U-58',
     # 'U-60', 'U-61', 'U-62', 'U-63', 'U-64', 'U-65', 'U-66', 'U-67', 'U-68',
     # 'U-70', 'U-71', 'U-72', 'U-73', 'U-74', 'U-75', 'U-76', 'U-77', 'U-78',
     # 'U-80', 'U-81', 'U-82', 'U-83', 'U-84', 'U-85', 'U-86', 'U-87', 'U-88',
]
for folder in folder_names:
    print(datetime.datetime.now().time())
    full_path_t = os.path.join(base_path, folder, file_pattern_t)
    full_path_dt = os.path.join(base_path, folder, file_pattern_dt)
    all_files_t = glob.glob(full_path_t)
    all_files_dt = glob.glob(full_path_dt)
    #####
    summer_files_t = [f for f in all_files_t if '-06-' in f or '-07-' in f or '-08-' in f or '-09-' in f]
    summer_files_dt = [f for f in all_files_dt if '-06-' in f or '-07-' in f or '-08-' in f or '-09-' in f]
    
    
    ds_t = xr.open_mfdataset(summer_files_t)
    ds_t = ds_t.sel(time=ds_t['time'].dt.month.isin([6, 7, 8]))
    ds_dt = xr.open_mfdataset(summer_files_dt)
    ds_dt = ds_dt.sel(time=ds_dt['time'].dt.month.isin([6, 7, 8]))    
    

    grouped_t = ds_t.groupby('time.year').groups
    grouped_dt = ds_dt.groupby('time.year').groups

    for year, year_indices_t in grouped_t.items():
        year_indices_t = grouped_t[year]
        year_indices_dt = grouped_dt[year]

        ds_year_t = ds_t.isel(time=year_indices_t)
        ds_year_dt = ds_dt.isel(time=year_indices_dt)

        monthly_groups_t = ds_year_t.groupby('time.month').groups
        monthly_groups_dt = ds_year_dt.groupby('time.month').groups

        for month, month_indices_t in monthly_groups_t.items():
            month_indices_t = monthly_groups_t[month]
            month_indices_dt = monthly_groups_dt[month]

            ds_month_t = ds_year_t.isel(time=month_indices_t)
            ds_month_dt = ds_year_dt.isel(time=month_indices_dt)

            daily_groups_t = ds_month_t.groupby('time.day').groups
            daily_groups_dt = ds_month_dt.groupby('time.day').groups

            # 循环遍历每日的数据
            for day, day_indices_t in daily_groups_t.items():
                day_indices_t = daily_groups_t[day]
                day_indices_dt = daily_groups_dt[day]

                ds_day_t = ds_month_t.isel(time=day_indices_t)
                ds_day_dt = ds_month_dt.isel(time=day_indices_dt)

                filename_t = f'{folder}_mt_{year}_{month:02d}_{day:02d}.nc'
                filename_dt = f'{folder}_mdt_{year}_{month:02d}_{day:02d}.nc'

                # 导出每日数据为 NetCDF 文件
                ds_day_t.to_netcdf(output+filename_t)
                ds_day_dt.to_netcdf(output+filename_dt)
