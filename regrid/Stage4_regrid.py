import xarray as xr
import xesmf as xe
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import datetime

pattern = '/N/project/Zli_lab/gongg/stage4_data/stage4_nc_re/*.nc'
files = sorted(glob.glob(pattern))

ds_ref1 = xr.open_mfdataset("ref_04deg.nc")
ds_ref2 = xr.open_mfdataset("ref_.1deg.nc")

test_ds_wrf1 = xr.open_dataset('/N/project/Zli_lab/gongg/stage4_data/stage4_nc_re/2002060100.nc')
test_ds_wrf2 = xr.open_dataset('/N/project/Zli_lab/gongg/stage4_data/regrid_bl/20020601.nc')

regridder1 = xe.Regridder(test_ds_wrf1, ds_ref1, method="bilinear", periodic=False)
regridder2 = xe.Regridder(test_ds_wrf2, ds_ref2, method="conservative", periodic=False)


chunk_size = 24
for i in range(0, len(files), chunk_size): # 100 should be len(files_jja)
    chunk = files[i : i + chunk_size]
    
    ds_wrf = xr.open_mfdataset(chunk, combine='by_coords')

    ds_wrf_regrid = regridder2(regridder1(ds_wrf))
    
    yyyymmdd = os.path.basename(chunk[0])[0:8]                           

    out_file = f'/N/project/Zli_lab/gongg/stage4_data/regrid_cons/{yyyymmdd}.nc'
    
    ds_wrf_regrid.to_netcdf(out_file)