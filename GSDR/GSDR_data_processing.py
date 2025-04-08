from datetime import datetime, timedelta


def local_to_utc(start_datetime_str, timezone_offset):
    local_time = datetime.strptime(start_datetime_str, '%Y%m%d%H')
    offset = timedelta(hours=timezone_offset)
    utc_time = local_time - offset
    utc_time_str = utc_time.strftime('%Y%m%d%H')
    
    return utc_time_str

def calculate_lst(utc_time_str, longitude):
    utc_time = datetime.strptime(utc_time_str, '%Y%m%d%H')
    time_difference = longitude / 15.0  
    lst_time = utc_time + timedelta(hours=time_difference)

    minutes = lst_time.minute
    if minutes < 30:
        lst_time = lst_time - timedelta(minutes=minutes)
    else:
        lst_time = lst_time + timedelta(minutes=(60 - minutes))
    
    lst_time_str = lst_time.strftime('%Y%m%d%H')
    
    return lst_time_str


import pandas as pd
import re
import os  # Ensure this import is included for os.listdir

directory = '/N/project/Zli_lab/Data/GSDR/QC_d_data_US/'

time_range = pd.date_range(start='1900-01-01 00:00', end='2014-01-01 01:00', freq='H')
time_list = time_range.strftime('%Y%m%d%H').tolist()

df = pd.DataFrame({'datetime': time_list})
file_count = 0
batch_size = 500
batch_number = 0

for filename in os.listdir(directory):
    if filename.endswith('.txt'):

        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()

            # Extract Station ID
            station_id = lines[0].split(':')[1].strip()

            # Extract Time Zone and clean it up
            time_zone_str = lines[16].split(':')[1].strip()
            match = re.search(r'\(UTC[^\)]+\)', time_zone_str)
            if match:
                time_zone = match.group(0).strip('()')
            else:
                time_zone = 'Unknown'

            # Extract Latitude and Longitude
            latitude = lines[5].split(':')[1].strip()
            longitude = lines[6].split(':')[1].strip()

            # Extract Start Datetime
            start_datetime = lines[7].split(':')[1].strip()
            
            if match.group(0).strip('()')[-1] in ['5', '6', '7', '8']:

                # Ensure calculate_lst and local_to_utc functions are defined
                lst_time_str = calculate_lst(local_to_utc(start_datetime, -int(match.group(0).strip('()')[-1])), float(longitude))

                # Process Precipitation Data
                precip_data = lines[21:]
                precip_values = [float(value.strip()) if value.strip() != '-999' else None for value in precip_data]

                try:
                    start_index = time_list.index(lst_time_str)
                except ValueError:
                    print(f"lst datetime {lst_time_str} not found in time_list.")
                    continue

                aligned_precip_values = [None] * len(time_list)
                for i, value in enumerate(precip_values):
                    if start_index + i < len(time_list):
                        aligned_precip_values[start_index + i] = value

                # Format the column name
                column_name = f"{station_id}, {time_zone}, {longitude}, {latitude}"
                df[column_name] = aligned_precip_values
                
            else:
                continue

            file_count += 1

            # Every 500 files, save a new CSV file
            if file_count % batch_size == 0:
                batch_number += 1
                output_filename = f'../GSDR_data/sumup_GSDR{(batch_number-1)*batch_size}-{batch_number*batch_size}.csv'
                df.to_csv(output_filename, index=False)
                print(f"{file_count} files processed. Saved as {output_filename}")
                
                # Reset dataframe for the next batch
                df = pd.DataFrame({'datetime': time_list})

# Save remaining data if there are leftover files
if not df.empty:
    batch_number += 1
    output_filename = f'sumup_GSDR{(batch_number-1)*batch_size}-{file_count}.csv'
    df.to_csv(output_filename, index=False)
    print(f"Processed remaining files. Saved as {output_filename}")





