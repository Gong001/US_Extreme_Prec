import os
import glob
import xarray as xr
from collections import defaultdict
import datetime


# 输入、输出目录
input_dir2 = '/N/project/Zli_lab/gongg/stage4_data/stage4_nc_re'
output_dir = '/N/project/Zli_lab/gongg/stage4_data/stage4_daily'
os.makedirs(output_dir, exist_ok=True)

# 获取所有 .nc 文件
grib_files2 = sorted(glob.glob(os.path.join(input_dir2, '*.nc')))
print(datetime.datetime.now())
# 将文件按“YYYYMMDD”分组
daily_files = defaultdict(list)
for filepath in grib_files2:
    # 假设文件名格式为 2002060100.nc => 前8位是YYYYMMDD
    filename = os.path.basename(filepath)  # 例如 "2002060100.nc"
    day_str = filename[:8]                # "20020601"
    daily_files[day_str].append(filepath)

# 逐天合并并保存
for day_str, file_list in list(daily_files.items())[1100:]:
    # 合并当日所有小时文件
    try:
        ds = xr.open_mfdataset(file_list, combine='by_coords', parallel=True)
        # 输出文件名如 20020601.nc
        out_filename = f"{day_str}.nc"
        out_path = os.path.join(output_dir, out_filename)
        
        # 保存
        ds.to_netcdf(out_path)
        
    except Exception as e:
        pass

print(datetime.datetime.now())



