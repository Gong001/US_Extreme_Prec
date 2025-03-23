#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
warnings.filterwarnings('ignore')
from pylab import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Wedge, Circle
import geopandas as gpd
from shapely.geometry import Point
import datetime
import glob


# In[2]:


def RxHhr(prec, latt, lonn, n):

    arr = prec.reshape(43, 2208, prec.shape[1], prec.shape[2])

    max_prec = np.full((43, prec.shape[1], prec.shape[2]), np.nan)

    for year in range(43):
        for i in range(prec.shape[1]):
            for j in range(prec.shape[2]):
                sliding_windows = np.lib.stride_tricks.sliding_window_view(arr[year, :, i, j], n)
                window_sums = np.sum(sliding_windows, axis=1)
                local_max = np.max(window_sums)

                if local_max > 1:
                    max_prec[year, i, j] = local_max

    ds_RxHhr = xr.Dataset(
        {'p': (['time', 'lat', 'lon'], max_prec)},
        coords={
            'time': (['time'], np.arange(1980, 2023)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )

    return ds_RxHhr



# In[3]:


def Rx1hrP(prec, latt, lonn):
    # Reshape the precipitation data to a 5-dimensional array (years, days, hours, lat, lon)
    arr = prec.reshape(43, 92, 24, prec.shape[1], prec.shape[2])
    
    # Initialize the output array for storing percentages
    arr_percent = np.full((43, 92, prec.shape[1], prec.shape[2]), np.nan)
    
    # Loop over years, days, and spatial dimensions
    for year in range(43):
        for day in range(92):
            for i in range(prec.shape[1]):  # Latitude
                for j in range(prec.shape[2]):  # Longitude
                    # Calculate the daily total precipitation using nansum to ignore NaNs
                    daily_total = np.sum(arr[year, day, :, i, j])
                    
                    # Calculate the maximum hourly precipitation using nanmax to ignore NaNs
                    daily_max = np.max(arr[year, day, :, i, j])
                    
                    # Calculate the percentage if daily total is not zero
                    if daily_total > 0:
                        arr_percent[year, day, i, j] = (daily_max / daily_total) 
                    if daily_total == 0:
                        arr_percent[year, day, i, j] = 0
    # Create an xarray dataset with the arr_percent data and appropriate coordinates
    ds_Rx1hrP = xr.Dataset(
        {'percent': (['year', 'day', 'lat', 'lon'], arr_percent)},
        coords={
            'year': (['year'], np.arange(1980, 2023)),
            'day': (['day'], np.arange(0, 92)),  # Assuming days are indexed from 1 to 92
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )
    
    return ds_Rx1hrP


# In[4]:


def RQpwHhrP(prec, latt, lonn):
    # Reshape the precipitation data
    arr = prec.reshape(43, 2208, prec.shape[1], prec.shape[2])

    # Calculate the mask based on NaN presence in the original data
    mask = np.isnan(np.nanmean(prec, axis=0))

    # Initialize output arrays
    percent_95 = np.full((43, prec.shape[1], prec.shape[2]), np.nan)
    percent_99 = np.full((43, prec.shape[1], prec.shape[2]), np.nan)
    quantile_95 = np.full((prec.shape[1], prec.shape[2]), np.nan)
    quantile_99 = np.full((prec.shape[1], prec.shape[2]), np.nan)

    # Processing data
    for year in range(43):
        for i in range(prec.shape[1]):
            for j in range(prec.shape[2]):
                # Extract yearly precipitation data
                yearly_precip = arr[year, :, i, j]

                # Filter wet hours
                wet_hours = yearly_precip[yearly_precip >= 0.1]

                # Calculate 95% and 99% quantiles
                q95 = np.percentile(wet_hours, 95) if len(wet_hours) > 0 else np.nan
                q99 = np.percentile(wet_hours, 99) if len(wet_hours) > 0 else np.nan
                
                quantile_95[i, j] = q95
                quantile_99[i, j] = q99
                
                # Total precipitation for wet hours
                total_wet_precip = np.sum(wet_hours)

                # Total exceeding 95% and 99% quantiles
                sum_over_q95 = np.sum(yearly_precip[yearly_precip >= q95])
                sum_over_q99 = np.sum(yearly_precip[yearly_precip >= q99])

                # Calculate percentages
                if total_wet_precip >= 0.1:
                    percent_95[year, i, j] = (sum_over_q95 / total_wet_precip)
                    percent_99[year, i, j] = (sum_over_q99 / total_wet_precip)
                else:
                    percent_95[year, i, j] = 0
                    percent_99[year, i, j] = 0

    # Apply mask
    percent_95[:, mask] = np.nan
    percent_99[:, mask] = np.nan

    # Create xarray datasets
    ds_percent = xr.Dataset(
        {
            'percent_95': (['year', 'lat', 'lon'], percent_95),
            'percent_99': (['year', 'lat', 'lon'], percent_99)
        },
        coords={
            'year': (['year'], np.arange(1980, 2023)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )
    ds_quantile = xr.Dataset(
        {
            'q_95': (['lat', 'lon'], quantile_95),
            'q_99': (['lat', 'lon'], quantile_99)
        },
        coords={
            'lon': (['lon'], lonn),
            'lat': (['lat'], latt),
        }
    )

    return ds_percent, ds_quantile


# In[5]:


def RHhrTmm(prec, latt, lonn, H):

    years = 43  # Assuming each year has 2208 hours
    # Reshape precipitation data to (years, hours, lat, lon)
    arr = prec.reshape(years, 2208, prec.shape[1], prec.shape[2])

    # Calculate the mask for NaN values in the original data
    mask = np.isnan(np.nanmean(prec, axis=0))

    # Initialize the result array
    arr_RHhrTmm_10 = np.zeros((years, prec.shape[1], prec.shape[2]), dtype=float)
    arr_RHhrTmm_20 = np.zeros((years, prec.shape[1], prec.shape[2]), dtype=float)
    arr_RHhrTmm_30 = np.zeros((years, prec.shape[1], prec.shape[2]), dtype=float)
    arr_RHhrTmm_50 = np.zeros((years, prec.shape[1], prec.shape[2]), dtype=float)
    # Process each year
    for year in range(years):
        for hour in range(0, 2208, H):
            # Calculate summed precipitation over the interval
            if hour + H <= 2208:
                summed_precip = np.nansum(arr[year, hour:hour + H, :, :], axis=0)
            else:
                summed_precip = np.nansum(arr[year, hour:2208, :, :], axis=0)

            # Check for exceedances over the threshold
            exceedances_10 = summed_precip >= 10
            exceedances_20 = summed_precip >= 20
            exceedances_30 = summed_precip >= 30
            exceedances_50 = summed_precip >= 50
            arr_RHhrTmm_10[year, :, :] += exceedances_10.astype(int)
            arr_RHhrTmm_20[year, :, :] += exceedances_20.astype(int)
            arr_RHhrTmm_30[year, :, :] += exceedances_30.astype(int)
            arr_RHhrTmm_50[year, :, :] += exceedances_50.astype(int)
            
            
    # Apply the mask to the result array
    arr_RHhrTmm_10[:, mask] = np.nan
    arr_RHhrTmm_20[:, mask] = np.nan
    arr_RHhrTmm_30[:, mask] = np.nan
    arr_RHhrTmm_50[:, mask] = np.nan
    # Create an xarray dataset to store the results
    ds_RHhrTmm = xr.Dataset(
        {
            'c_10': (['year', 'lat', 'lon'], arr_RHhrTmm_10),
            'c_20': (['year', 'lat', 'lon'], arr_RHhrTmm_20),
            'c_30': (['year', 'lat', 'lon'], arr_RHhrTmm_30),
            'c_50': (['year', 'lat', 'lon'], arr_RHhrTmm_50),
        },
        coords={
            'year': (['year'], np.arange(1980, 1980 + years)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )

    return ds_RHhrTmm


# In[6]:


def NWH(prec, latt, lonn):
    arr = prec.reshape(43, 2208, prec.shape[1], prec.shape[2])
    mask = np.isnan(np.nanmean(prec, axis=0))

    nwh = np.zeros((43, prec.shape[1], prec.shape[2]))

    for year in range(43):
        for i in range(prec.shape[1]):
            for j in range(prec.shape[2]):
                hourly_data = arr[year, :, i, j]
                wet_hours = hourly_data >= 0.1
                nwh[year, i, j] = np.sum(wet_hours)

    nwh[:, mask] = np.nan
    ds_NWH = xr.Dataset(
        {'c': (['year', 'lat', 'lon'], nwh)},
        coords={
            'year': (['year'], np.arange(1980, 2023)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )
    
    return ds_NWH


# In[7]:


def MeLWS_MxLWS(prec,latt,lonn):

    arr = prec.reshape(43, 2208, prec.shape[1], prec.shape[2])
    mask = np.isnan(np.nanmean(prec, axis=0))
    MeLWS = np.full((43, prec.shape[1], prec.shape[2]), np.nan)
    MxLWS = np.full((43, prec.shape[1], prec.shape[2]), np.nan)

    for year in range(43):
        for i in range(prec.shape[1]):
            for j in range(prec.shape[2]):
                yearly_precip = arr[year, :, i, j]
                is_wet = yearly_precip >= 0.1
                wet_starts = np.where(np.diff(is_wet.astype(int)) == 1)[0] + 1
                wet_ends = np.where(np.diff(is_wet.astype(int)) == -1)[0] + 1

                if is_wet[0]:
                    wet_starts = np.insert(wet_starts, 0, 0)
                if is_wet[-1]:
                    wet_ends = np.append(wet_ends, is_wet.size)

                if wet_starts.size > 0:  
                    wet_lengths = wet_ends - wet_starts
                    MeLWS[year, i, j] = np.mean(wet_lengths)
                    MxLWS[year, i, j] = np.max(wet_lengths)
                else:
                    MeLWS[year, i, j] = 0
                    MxLWS[year, i, j] = 0


    MeLWS[:, mask] = np.nan
    MxLWS[:, mask] = np.nan

    ds_MeLWS_MxLWS = xr.Dataset(
        {
            'MeLWS': (['year', 'lat', 'lon'], MeLWS),
            'MxLWS': (['year', 'lat', 'lon'], MxLWS)
        },
        coords={
            'year': (['year'], np.arange(1980, 2023)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )
    
    return ds_MeLWS_MxLWS


# In[8]:


def SPIIHhr(prec, latt, lonn, H):
    arr = prec.reshape(43,2208, prec.shape[1], prec.shape[2])
    spi_ihhr = np.full((43, prec.shape[1], prec.shape[2]), np.nan)

    for year in range(43):
        for i in range(prec.shape[1]):
            for j in range(prec.shape[2]):
                total_precip = []

                for hour in range(0, 2208, H):
                    if hour + H <= 2208:
                        precip_sum = np.sum(arr[year, hour:hour + H, i, j])
                    else:
                        precip_sum = np.sum(arr[year, hour:, i, j])

                    if precip_sum >= 0.1:
                        total_precip.append(precip_sum)

                if total_precip:
                    spi_ihhr[year, i, j] = np.mean(total_precip)

    ds_SPIIHhr = xr.Dataset(
        {'p': (['year', 'lat', 'lon'], spi_ihhr)},
        coords={
            'year': (['year'], np.arange(1980, 2023)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )
    
    return ds_SPIIHhr


# In[9]:


def RTot(prec,latt,lonn):
    arr = prec.reshape(43, 2208, prec.shape[1], prec.shape[2])
    mask = np.isnan(np.nanmean(prec, axis=0))
    new_arr = np.where(arr > 0.1, arr, np.nan)
    RTot = np.nansum(new_arr,axis=1)
    RTot[:, mask] = np.nan
    ds_RTot = xr.Dataset(
        {'p': (['year', 'lat', 'lon'], RTot)},
        coords={
            'year': (['year'], np.arange(1980, 2023)),
            'lat': (['lat'], latt),
            'lon': (['lon'], lonn)
        }
    )
    return ds_RTot


# In[ ]:





# In[33]:


base_path = '/N/project/Zli_lab/gongg/CONUS404_data/LST/UTC/'
file_pattern = 'PREC_ACC_NC.wrf2d_d01_????-??-??.nc'
output_folder = '/N/project/Zli_lab/gongg/CONUS404_data/LST/test/'
folder_names = [
  
    'U-50',
   #  'U-81', 'U-82', 'U-83', 'U-84', 'U-85', 'U-86', 'U-87', 'U-88'
]


for folder in folder_names:
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    full_path = os.path.join(base_path, folder, file_pattern)
    all_files = glob.glob(full_path)
    #####
    summer_files = [f for f in all_files if '-06-' in f or '-07-' in f or '-08-' in f or '-09-' in f]
    ds_summer = xr.open_mfdataset(summer_files)
    ds_jja = ds_summer.sel(time=ds_summer['time'].dt.month.isin([6, 7, 8]))
    lonn = ds_jja.lon.values
    latt = ds_jja.lat.values
    prec = ds_jja.p.values
    
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_RxHhr_1 = RxHhr(prec, latt, lonn, 1)
    ds_RxHhr_3 = RxHhr(prec, latt, lonn, 3)
    ds_RxHhr_6 = RxHhr(prec, latt, lonn, 6)
    ds_RxHhr_12 = RxHhr(prec, latt, lonn, 12)
    ds_RxHhr_24 = RxHhr(prec, latt, lonn, 24)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_Rx1hrP = Rx1hrP(prec, latt, lonn)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_percent, ds_quantile = RQpwHhrP(prec, latt, lonn)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_RHhrTmm_1 = RHhrTmm(prec, latt, lonn, 1)
    ds_RHhrTmm_3 = RHhrTmm(prec, latt, lonn, 3)
    ds_RHhrTmm_6 = RHhrTmm(prec, latt, lonn, 6)
    ds_RHhrTmm_12 = RHhrTmm(prec, latt, lonn, 12)
    ds_RHhrTmm_24 = RHhrTmm(prec, latt, lonn, 24)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_NWH = NWH(prec, latt, lonn)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_MeLWS_MxLWS = MeLWS_MxLWS(prec,latt,lonn)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_SPIIHhr_1 = SPIIHhr(prec, latt, lonn, 1)
    ds_SPIIHhr_3 = SPIIHhr(prec, latt, lonn, 3)
    ds_SPIIHhr_6 = SPIIHhr(prec, latt, lonn, 6)
    ds_SPIIHhr_12 = SPIIHhr(prec, latt, lonn, 12)
    ds_SPIIHhr_24 = SPIIHhr(prec, latt, lonn, 24)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_RTot = RTot(prec,latt,lonn)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    
    ################################################################################################
    
    
    ds_RxHhr_1.to_netcdf(output_folder+'ds_RxHhr_1_'+folder+'.nc')
    ds_RxHhr_3.to_netcdf(output_folder+'ds_RxHhr_3_'+folder+'.nc')
    ds_RxHhr_6.to_netcdf(output_folder+'ds_RxHhr_6_'+folder+'.nc')
    ds_RxHhr_12.to_netcdf(output_folder+'ds_RxHhr_12_'+folder+'.nc')
    ds_RxHhr_24.to_netcdf(output_folder+'ds_RxHhr_24_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_Rx1hrP.to_netcdf(output_folder+'ds_Rx1hrP_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_percent.to_netcdf(output_folder+'ds_percent_'+folder+'.nc')
    ds_quantile.to_netcdf(output_folder+'ds_quantile_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_RHhrTmm_1.to_netcdf(output_folder+'ds_RHhrTmm_1_'+folder+'.nc')
    ds_RHhrTmm_3.to_netcdf(output_folder+'ds_RHhrTmm_3_'+folder+'.nc')
    ds_RHhrTmm_6.to_netcdf(output_folder+'ds_RHhrTmm_6_'+folder+'.nc')
    ds_RHhrTmm_12.to_netcdf(output_folder+'ds_RHhrTmm_12_'+folder+'.nc')
    ds_RHhrTmm_24.to_netcdf(output_folder+'ds_RHhrTmm_24_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_NWH.to_netcdf(output_folder+'ds_NWH_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_MeLWS_MxLWS.to_netcdf(output_folder+'ds_MeLWS_MxLWS_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_SPIIHhr_1.to_netcdf(output_folder+'ds_SPIIHhr_1_'+folder+'.nc')
    ds_SPIIHhr_3.to_netcdf(output_folder+'ds_SPIIHhr_3_'+folder+'.nc')
    ds_SPIIHhr_6.to_netcdf(output_folder+'ds_SPIIHhr_6_'+folder+'.nc')
    ds_SPIIHhr_12.to_netcdf(output_folder+'ds_SPIIHhr_12_'+folder+'.nc')
    ds_SPIIHhr_24.to_netcdf(output_folder+'ds_SPIIHhr_24_'+folder+'.nc')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    ds_RTot.to_netcdf(output_folder+'ds_RTot_'+folder+'.nc')
    




