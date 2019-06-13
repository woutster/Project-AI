import pandas as pd
import numpy as np
import ast
import geopy.distance

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
    point1 = ast.literal_eval(point1)
    point2 = ast.literal_eval(point2)
    return geopy.distance.vincenty(point1, point2).m


def calculate_cmf(speed, distance):
    speed_squared = np.square(speed)
    diff = np.diff(speed_squared)
    cmf_pos = np.sum(np.maximum(diff, 0))/distance
    cmf_neg = np.sum(np.maximum(-diff, 0))/distance
    return cmf_pos, cmf_neg


def add_speeds(df, df_fenced):
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

    df_fenced.to_csv('process_speed_proov_00' +file_name+ '.csv', sep=';', index=False)
    return df_fenced


def merge_data(df_speed, df_combined, idx, file_name):
    df_combined.drop(idx, inplace=True)

    df_combined = df_combined.set_index('in_time')
    df_speed = df_speed.set_index('timestamp_entry')

    df_merged = pd.concat([df_speed, df_combined], axis=1)
    df_merged['hour_of_day'] = df_merged['timestamp_exit'].dt.round('H').dt.hour

    df_merged.drop(['in_lat', 'in_long', 'out_lat', 'out_long', 'out_time', 'date', 'day', 'time'], axis=1, inplace=True)
    df_fenced.to_csv('Data/proov_00' + file_name + '/proov_00' + file_name + '_merged_data.csv', sep=';')
    return df_merged


if __name__ == '__main__':
    for i in range(1,4):
        file_name = str(i)
        print("Process file proov_00" + file_name + ".")
        # df = pd.read_csv('Data/proov_00' + file_name + '/speed.csv')
        df_combined = pd.read_csv(filepath_or_buffer='Data/proov_00' + file_name + '/combined.csv', sep=',',
                                  parse_dates=['in_time', 'out_time'])
        df_fenced, idx = preproc_gps_data(pd.read_csv('Data/proov_00' + file_name + '/proov_00' + file_name + '_geoFenced.csv', header=None))
        #
        # df['time'] = pd.to_datetime(df['time'], unit='ms')
        # df.set_index('time', inplace=True)
        #
        # df_speed = add_speeds(df, df_fenced)

        df_speed = pd.read_csv(filepath_or_buffer='Data/proov_00' + file_name + '/process_speed_proov_00' + file_name + '.csv', sep=';',
                               parse_dates=['timestamp_entry', 'timestamp_exit'])
        df_speed['timestamp_entry'] = df_speed['timestamp_entry'].dt.round('1s')
        df_speed['timestamp_exit'] = df_speed['timestamp_exit'].dt.round('1s')

        df_merged = merge_data(df_speed, df_combined, idx, file_name)