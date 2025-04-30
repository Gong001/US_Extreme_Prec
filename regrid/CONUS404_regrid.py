import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime



pattern_june = '/N/project/Zli_lab/Data/Observations/NCAR/prec_acc_files/PREC_ACC_NC.wrf2d_d01_????-06-??_*.nc'
pattern_july = '/N/project/Zli_lab/Data/Observations/NCAR/prec_acc_files/PREC_ACC_NC.wrf2d_d01_????-07-??_*.nc'
pattern_aug  = '/N/project/Zli_lab/Data/Observations/NCAR/prec_acc_files/PREC_ACC_NC.wrf2d_d01_????-08-??_*.nc'
pattern_spet  = '/N/project/Zli_lab/Data/Observations/NCAR/prec_acc_files/PREC_ACC_NC.wrf2d_d01_????-09-??_*.nc'


files_june = glob.glob(pattern_june)
files_july = glob.glob(pattern_july)
files_aug  = glob.glob(pattern_aug)
files_spet  = glob.glob(pattern_spet)

files_jja = sorted(files_june + files_july + files_aug + files_spet)

ds_ref1 = xr.open_mfdataset("ref_04deg.nc")
ds_ref2 = xr.open_mfdataset("ref_.1deg.nc")

test_ds_wrf1 = xr.open_dataset('/N/project/Zli_lab/Data/Observations/NCAR/prec_acc_files/PREC_ACC_NC.wrf2d_d01_1980-06-01_00:00:00.nc')
test_ds_wrf2 = xr.open_dataset('/N/project/Zli_lab/gongg/CONUS404_data/regrid_bl/19800601.nc')


regridder1 = xe.Regridder(test_ds_wrf1, ds_ref1, method="bilinear", periodic=False)
regridder2 = xe.Regridder(test_ds_wrf2, ds_ref2, method="conservative", periodic=False)


chunk_size = 24
for i in range(0, len(files_jja), chunk_size): # 100 should be len(files_jja)
    chunk = files_jja[i : i + chunk_size]
    
    ds_wrf = xr.open_mfdataset(chunk, combine='by_coords')

    ds_wrf_regrid = regridder2(regridder1(ds_wrf))
    
    basename = os.path.basename(chunk[0])               
    date_str = basename.split("_")[4]                  
    yyyymmdd = date_str.replace("-", "")                

    out_file = f'/N/project/Zli_lab/gongg/CONUS404_data/regrid_cons/{yyyymmdd}.nc'

    ds_cleaned = ds_wrf_regrid.drop_vars('XTIME')
    ds_cleaned = ds_cleaned.rename({'Time': 'time'})
    
    ds_cleaned.to_netcdf(out_file)

