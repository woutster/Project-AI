import pandas as pd
import numpy as np
import ast
import geopy.distance
import csv
import requests




start_date = '2017-01-02'
end_date = '2017-01-02'
set_hour = 7

file_name = '1'

# Timestep to calculate if vehicle is accelerating/braking in seconds
delta_accel = 4
delta_timestep = 1

# Disable copy warning
pd.options.mode.chained_assignment = None


def fix_timesteps(df):
    """ Changes the timestemps from miliseconds to a datetime property."""

    orig_len = len(df)
    df['timestamp_entry'] = pd.to_datetime(df['timestamp_entry'],
                                                  format='Timestamp(\'%Y-%m-%d %H:%M:%S.%f\')', errors='coerce')
    idx_in = df[df['timestamp_entry'].isnull()].index

    df.drop(idx_in, inplace=True)
    df['timestamp_exit'] = pd.to_datetime(df['timestamp_exit'],
                                           format='Timestamp(\'%Y-%m-%d %H:%M:%S.%f\')', errors='coerce')
    idx_out = df[df['timestamp_exit'].isnull()].index
    df.drop(idx_out, inplace=True)
    idx = idx_in.append(idx_out)
    print("Dropping", len(idx), "out of", orig_len, "datapoints because of wrong format.")
    return df, idx


def preproc_gps_data(df):
    """ Reformats data so it's easier to work with."""

    df[['1_', '2_', '3_']] = df[0].str.split(',', expand=True)
    df[['4_', '5_', '6_']] = df[1].str.split(',', expand=True)
    df['gps_point_entry'] = (df['1_'] + ',' + df['2_']).str.replace(r'^\[', '')
    df['timestamp_entry'] = (df['3_']).str.replace(r'\]$', '').str.strip()
    df['gps_point_exit'] = (df['4_'] + ',' + df['5_']).str.replace(r'^\[', '')
    df['timestamp_exit'] = (df['6_']).str.replace(r'\]$', '').str.strip()
    df.drop([0, 1, '1_', '2_', '3_', '4_', '5_', '6_'], axis=1, inplace=True)
    df, idx = fix_timesteps(df)
    return df, idx


def calculate_distance(point1, point2):
    """ Calculates distance from point A to point B. """
    point1 = ast.literal_eval(point1)
    point2 = ast.literal_eval(point2)
    return geopy.distance.vincenty(point1, point2).m


def calculate_cmf(speed, distance):
    """ Calculates the positive and negative constant motion factor.
    Args:
        speed: speed of Bus
        distance: distance a bus has traveled
    Returns:
        cmf_pos: positive cmf
        cmf_neg: negative cmf
    """
    speed_squared = np.square(speed)
    diff = np.diff(speed_squared)
    cmf_pos = np.sum(np.maximum(diff, 0))/distance
    cmf_neg = np.sum(np.maximum(-diff, 0))/distance
    return cmf_pos, cmf_neg


def add_speeds(df, df_fenced):
    """ Adds speed data to fenced DataFrame"""

    counter = 0
    all_speeds = {}
    # df_fenced = df_fenced[:11]
    total_rows = df_fenced.shape[0]

    for index, row in df_fenced.iterrows():

        timestamp_in = row['timestamp_entry']
        timestamp_out = row['timestamp_exit']

        timestamp_df = df[(df.index >= timestamp_in) & (df.index <= timestamp_out)]

        speeds_in_fence = timestamp_df['speed'].values

        len_list = len(speeds_in_fence)
        # Certain amount of points is needed to calculate cmf, so will be ignored if too little points.
        if len_list >= 5:
            distance = calculate_distance(row['gps_point_entry'], row['gps_point_exit'])
            cmf_pos, cmf_neg = calculate_cmf(speeds_in_fence, distance)
            all_speeds[timestamp_in] = [np.average(speeds_in_fence), cmf_pos, cmf_neg]

        else:
            print("this pass only has", len_list, "instances in the fence and we deem that too few for actual measures...")
            all_speeds[timestamp_in] = [np.nan, np.nan, np.nan]

        counter += 1
        print('Processed row', counter, '/', total_rows)

    speed_df = pd.DataFrame.from_dict(all_speeds, orient='index').values
    df_fenced['avg_speed'] = pd.DataFrame(speed_df[:, 0])
    df_fenced['cmf_pos'] = pd.DataFrame(speed_df[:, 1])
    df_fenced['cmf_neg'] = pd.DataFrame(speed_df[:, 2])

    # df_fenced.to_csv('process_speed_proov_00' +file_name+ '.csv', sep=';', index=False)
    return df_fenced

def csv_to_pd(filename):
    """ Read in csv file with geofenced data and return dataframe. """

    df = pd.read_csv(filename, names=['in', 'out'])

    # Make file pretty
    df.replace('\[', '', regex=True, inplace=True)
    df.replace('\]', '', regex=True, inplace=True)
    df.replace('\(', '', regex=True, inplace=True)
    df.replace('\)', '', regex=True, inplace=True)
    df.replace('Timestamp', '', regex=True, inplace=True)
    df.replace("'", '', regex=True, inplace=True)
    df.replace(',', '', regex=True, inplace=True)

    # Split into neat columns for entering geofence
    new = df['in'].str.split(' ', n=2, expand=True)
    df['in_lat'] = new[0]
    df['in_long'] = new[1]
    df['in_time'] = pd.to_datetime(new[2], format='%Y-%m-%d %H:%M:%S')
    df['in_time'] = df['in_time'].dt.round("S")

    # Split into neat columns for exiting geofence
    new = df['out'].str.split(' ', n=2, expand=True)
    df['out_lat'] = new[0]
    df['out_long'] = new[1]
    df['out_time'] = pd.to_datetime(new[2], format='%Y-%m-%d %H:%M:%S')
    df['out_time'] = df['out_time'].dt.round("S")

    df['date'] = df['in_time'].dt.date
    df['day'] = df['in_time'].dt.dayofweek

    # 0 is Monday, 6 is Sunday
    one_hot_days = pd.get_dummies(df['day'])
    return df, one_hot_days


def weather_csv_to_pd(filename):
    """ Read in csv with weather data and return a dataframe. """

    df = pd.read_csv(filename, names=['time', 'temp', 'precipitation_intensity', 'wind_speed', 'visibility'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['time'] = df['time'].dt.round("S")
    df['date'] = df['time'].dt.date

    return df

def pd_to_csv(filename, geo, weather, one_hot_days):
    """Writes the dataframes with geo info, weather data, and one hot days
    of the week to csv file.
    """

    # Sort weather df.
    weather = weather.sort_values(by='date')
    weather.reset_index()

    with open(filename, mode='w') as csv_file:
        fieldnames = ['in_lat', 'in_long', 'in_time', 'out_lat', 'out_long', 'out_time',
        'date', 'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'holiday', 'time', 'temp', 'precipitation_intensity', 'wind_speed',
        'visibility']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, g in geo.iterrows():

            # Find proper weather data
            temp, perc, wind, vis, time = 0, 0, 0, 0, 0
            for j, w in weather.iterrows():
                if g['in_time'] <= w['time'] or g['out_time'] <= w['time'] :
                    temp, perc, wind, vis = w['temp'], w['precipitation_intensity'], w['wind_speed'], w['visibility']
                    time = w['time']
                    break
                elif g['date'] == w['date']:
                    temp, perc, wind, vis = w['temp'], w['precipitation_intensity'], w['wind_speed'], w['visibility']
                    time = w['time']
                    break

            # Create one hot vector of days
            # one_hot_day_vec = [one_hot_days[0].iloc[i], one_hot_days[1].iloc[i], one_hot_days[2].iloc[i], one_hot_days[3].iloc[i], one_hot_days[4].iloc[i], one_hot_days[5].iloc[i], one_hot_days[6].iloc[i]]

            writer.writerow({'in_lat': g['in_lat'],
                                'in_long': g['in_long'],
                                'in_time': g['in_time'],
                                'out_lat': g['out_lat'],
                                'out_long': g['out_long'],
                                'out_time': g['out_time'],
                                'date': g['date'],
                                'day': g['day'],
                                'monday': one_hot_days[0].iloc[i],
                                'tuesday': one_hot_days[1].iloc[i],
                                'wednesday': one_hot_days[2].iloc[i],
                                'thursday': one_hot_days[3].iloc[i],
                                'friday': one_hot_days[4].iloc[i],
                                'saturday': one_hot_days[5].iloc[i],
                                'sunday': one_hot_days[6].iloc[i],
                                'holiday': g['holiday'],
                                'time': time,
                                'temp': temp,
                                'precipitation_intensity': perc,
                                'wind_speed': wind,
                                'visibility': vis
                            })


def holidays(geo):
    """ Retrieves holiday dates from 2017 and 2018. """

    base_url = 'https://calendarific.com/api/v2/holidays?'

    api_key = '12bc7e5bac025c16e1daf74e12f8811c8460aec5'
    year = '2018'
    country = 'NL'
    url18 = f'{base_url}&api_key={api_key}&country={country}&year={year}'
    url17 = f'{base_url}&api_key={api_key}&country={country}&year=2017'

    response_2018 = requests.get(url18)
    response_2017 = requests.get(url17)
    data18 = response_2018.json()
    data17 = response_2017.json()

    geo['holiday'] = 0

    holidates = []
    # Reformat string
    for hol in data18['response']['holidays']:
        if int(hol['date']['datetime']['day']) < 10:
            hol['date']['datetime']['day'] = '0' + str(hol['date']['datetime']['day'])
        if int(hol['date']['datetime']['month']) < 10:
            hol['date']['datetime']['month'] = '0' + str(hol['date']['datetime']['month'])

        date = f"{hol['date']['datetime']['year']}-{hol['date']['datetime']['month']}-{hol['date']['datetime']['day']}"
        holidates.append(date)
    for hol in data17['response']['holidays']:
        if int(hol['date']['datetime']['day']) < 10:
            hol['date']['datetime']['day'] = '0' + str(hol['date']['datetime']['day'])
        if int(hol['date']['datetime']['month']) < 10:
            hol['date']['datetime']['month'] = '0' + str(hol['date']['datetime']['month'])

        date = f"{hol['date']['datetime']['year']}-{hol['date']['datetime']['month']}-{hol['date']['datetime']['day']}"
        holidates.append(date)

    # Create list whether days are holidays
    holiday_boolean_list = []
    for i, g in geo.iterrows():
        if str(g['date']) in holidates:
            holiday_boolean_list.append(1)
        else:
            holiday_boolean_list.append(0)

    # Add holidays to dataframe
    geo['holiday'] = holiday_boolean_list

    return geo


def merge_data(df_speed, df_combined, idx, file_name):
    """ Merges data from different DataFrames into one DataFrame. """
    df_combined.drop(idx, inplace=True)

    df_combined = df_combined.set_index('in_time')
    df_speed = df_speed.set_index('timestamp_entry')

    df_merged = pd.concat([df_speed, df_combined], axis=1)
    df_merged['hour_of_day'] = df_merged['timestamp_exit'].dt.round('H').dt.hour

    df_merged.drop(['in_lat', 'in_long', 'out_lat', 'out_long', 'out_time', 'date', 'day', 'time'], axis=1, inplace=True)
    df_merged.to_csv('Data/proov_00' + file_name + '/proov_00' + file_name + '_merged_data.csv', sep=';')
    return df_merged

def run_process_speed():
    """ Calls all needed functions for this part of the program. """
    for i in range(1,4):
        bus = 'proov_00' + str(i)
        print(bus)

        # Get geo info + days of week one hot encoded
        geo, one_hot_days = csv_to_pd(f'Data/{bus}/{bus}_geoFenced.csv')
        weather = weather_csv_to_pd(f'Data/{bus}/weather.csv')

        # Add holidays
        geo = holidays(geo)

        # Write to csv
        pd_to_csv(f'Data/{bus}/{bus}_combined.csv', geo, weather, one_hot_days)

        # file_name = str(i)
        print("Process file proov_00" + file_name + ".")
        # df = pd.read_csv('Data/proov_00' + file_name + '/speed.csv')

        # Read combined file
        df_combined = pd.read_csv(filepath_or_buffer=f'Data/{bus}/{bus}_combined.csv', sep=',',
                                  parse_dates=['in_time', 'out_time'])
        df_fenced, idx = preproc_gps_data(pd.read_csv(f'Data/{bus}/{bus}_geoFenced.csv', header=None))
        #
        # df['time'] = pd.to_datetime(df['time'], unit='ms')
        # df.set_index('time', inplace=True)
        #
        df_speed = add_speeds(df, df_fenced)

        # df_speed = pd.read_csv(filepath_or_buffer='Data/proov_00' + file_name + '/process_speed_proov_00' + file_name + '.csv', sep=';',
                               # parse_dates=['timestamp_entry', 'timestamp_exit'])
        df_speed['timestamp_entry'] = df_speed['timestamp_entry'].dt.round('1s')
        df_speed['timestamp_exit'] = df_speed['timestamp_exit'].dt.round('1s')

        df_merged = merge_data(df_speed, df_combined, idx, file_name)
