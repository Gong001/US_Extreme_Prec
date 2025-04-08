

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
gdf = gpd.read_file('/N/project/Zli_lab/Data/Other/tl_2019_us_state/tl_2019_us_state.shp')
US = gpd.read_file('/N/project/Zli_lab/Data/Other/tl_2019_us_state/tl_2019_us_state.shp')
base_path = '/N/project/Zli_lab/gongg/CONUS404_data/LST/JJA_dailydata'
output1 = '/N/project/Zli_lab/gongg/CONUS404_data/LST/'
ds_raster = xr.open_dataset('/N/project/Zli_lab/Data/Observations/NCAR/prec_acc_files/PREC_ACC_NC.wrf2d_d01_2022-09-30_23:00:00.nc')
# 定义所有的U*前缀
prefixes = [
    'U-50', 'U-51', 'U-52', 'U-53', 'U-54', 'U-55', 'U-56', 'U-57', 'U-58',
    'U-60', 'U-61', 'U-62', 'U-63', 'U-64', 'U-65', 'U-66', 'U-67', 'U-68',
    'U-70', 'U-71', 'U-72', 'U-73', 'U-74', 'U-75', 'U-76', 'U-77', 'U-78',
    'U-80', 'U-81', 'U-82', 'U-83', 'U-84', 'U-85', 'U-86', 'U-87', 'U-88'
]

for year in range(1990, 2001):  # 从1980年到2022年
    for month in [6, 7, 8]:  # 只读取6, 7, 8月的数据
        print(datetime.datetime.now().time())
        days_in_month = 30 if month == 6 else 31  # 6月30天，7月和8月31天
        
        
        for day in range(1, days_in_month + 1):
            files_to_open = []
            # 对每一个前缀和日期组合构造文件路径
            for prefix in prefixes:
                file_pattern = f'{base_path}/{prefix}_mdt_{year}_{month:02d}_{day:02d}.nc'
                files_to_open.append(file_pattern)
                
            ds = xr.open_mfdataset(files_to_open)
            lon = ds_raster['XLONG'].values
            lat = ds_raster['XLAT'].values
            grid = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(lon.flatten(), lat.flatten()),
                index=np.arange(lon.size)
            )
            grid.set_crs(gdf.crs, inplace=True)
            grid_s = gpd.sjoin(grid, gdf, how='inner', predicate='within')

            mask = np.full(ds_raster['PREC_ACC_NC'].shape[1:], False) 
            for index in grid_s.index:
                row, col = np.unravel_index(index, mask.shape)  # 获取行列索引
                mask[row, col] = True
            mask_da = xr.DataArray(mask, dims=ds_raster['PREC_ACC_NC'].dims[1:], coords={'south_north': ds_raster['PREC_ACC_NC'].coords['south_north'], 'west_east': ds_raster['PREC_ACC_NC'].coords['west_east']})
            ds_conus = ds_raster.where(mask_da, drop=True)

            XLON = ds_conus.XLONG.values[:707,:]
            XLAT = ds_conus.XLAT.values[:707,:]
            ds_n = ds.assign_coords({
                'XLON': (('lat', 'lon'), XLON),
                'XLAT': (('lat', 'lon'), XLAT)
            })

            regions_dict = {
                'NE': ['CT', 'DE', 'ME', 'MD', 'MA', 'NH', 'NJ', 'NY', 'PA', 'RI', 'VT', 'WV'],
                'Midwest': ['IA', 'MI', 'MN', 'WI', 'IL', 'IN', 'MO', 'OH'],
                'SE': ['AL', 'FL', 'GA', 'NC', 'SC', 'VA', 'TN', 'KY', 'AR', 'LA', 'MS'],
                'NGP': ['MT', 'NE', 'ND', 'SD', 'WY'],
                'SGP': ['KS', 'OK', 'TX'],
                'SW': ['AZ', 'CO', 'NM', 'UT', 'CA', 'NV'],
                'NW': ['ID', 'OR', 'WA']
            }
            regions = {name: US[US['STUSPS'].isin(states)] for name, states in regions_dict.items()}
            regi = ['NE','Midwest','SE','NGP','SGP','SW','NW',]

            ds_results = {}

            for r in regi:
                lon = ds_n['XLON'].values
                lat = ds_n['XLAT'].values
                grid = gpd.GeoDataFrame(
                    geometry=gpd.points_from_xy(lon.flatten(), lat.flatten()),
                    index=np.arange(lon.size)
                )

                grid.set_crs(regions[r].crs, inplace=True)
                grid_s = gpd.sjoin(grid, regions[r], how='inner', predicate='within')

                mask = np.full((ds_n['td2'].shape[1], ds_n['td2'].shape[2]), False) 
                for index in grid_s.index:
                    row, col = np.unravel_index(index, mask.shape)
                    mask[row, col] = True 

                mask_da = xr.DataArray(
                    mask, 
                    dims=['lat', 'lon'],
                    coords={
                        'lat': ds_n['lat'].values,
                        'lon': ds_n['lon'].values
                    }
                )

                ds_ssss = ds_n.where(mask_da, drop=True)
                ds_results[f'ds_{r}'] = ds_ssss

            ds_results['ds_NE'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}NE/mdt_{year}_{month:02d}_{day:02d}.nc")
            ds_results['ds_Midwest'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}Midwest/mdt_{year}_{month:02d}_{day:02d}.nc")
            ds_results['ds_SE'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}SE/mdt_{year}_{month:02d}_{day:02d}.nc")
            ds_results['ds_NGP'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}NGP/mdt_{year}_{month:02d}_{day:02d}.nc")
            ds_results['ds_SGP'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}SGP/mdt_{year}_{month:02d}_{day:02d}.nc")
            ds_results['ds_SW'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}SW/mdt_{year}_{month:02d}_{day:02d}.nc")
            ds_results['ds_NW'].drop_vars(['XLON', 'XLAT']).to_netcdf(f"{output1}NW/mdt_{year}_{month:02d}_{day:02d}.nc")