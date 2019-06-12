import pandas as pd
import numpy as np
import ast
import geopy.distance
import matplotlib.pyplot as plt

start_date = '2017-01-02'
end_date = '2017-01-02'
set_hour = 7

# Timestep to calculate if vehicle is accelerating/braking in seconds
delta_accel = 4
delta_timestep = 1


def preproc_gps_data(df):
    df[['1_', '2_', '3_']] = df[0].str.split(',', expand=True)
    df[['4_', '5_', '6_']] = df[1].str.split(',', expand=True)
    df['gps_point_entry'] = (df['1_'] + ',' + df['2_']).str.replace(r'^\[', '')
    df['timestamp_entry'] = (df['3_']).str.replace(r'\]$', '').str.strip()
    df['gps_point_exit'] = (df['4_'] + ',' + df['5_']).str.replace(r'^\[', '')
    df['timestamp_exit'] = (df['6_']).str.replace(r'\]$', '').str.strip()
    df['speeds'] = 0.
    df.drop([0, 1, '1_', '2_', '3_', '4_', '5_', '6_'], axis=1, inplace=True)
    return df


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
    speeds_in_fence = []
    all_speeds = []
    total_rows = df_fenced.shape[0]
    for index, row in df.iterrows():

        if counter == total_rows:
            break

        try:
            timestamp_in = pd.to_datetime(df_fenced['timestamp_entry'][counter],
                                          format='Timestamp(\'%Y-%m-%d %H:%M:%S.%f\')')
        except ValueError:
            timestamp_in = pd.to_datetime(df_fenced['timestamp_entry'][counter],
                                          format='Timestamp(\'%Y-%m-%d %H:%M:%S\')')
        try:
            timestamp_out = pd.to_datetime(df_fenced['timestamp_exit'][counter],
                                           format='Timestamp(\'%Y-%m-%d %H:%M:%S.%f\')')
        except ValueError:
            timestamp_out = pd.to_datetime(df_fenced['timestamp_exit'][counter],
                                           format='Timestamp(\'%Y-%m-%d %H:%M:%S\')')

        if index >= timestamp_in:
            speed = row['speed']
            speeds_in_fence.append(speed)

            if index >= timestamp_out:
                distance = calculate_distance(df_fenced.iloc[counter]['gps_point_entry'], df_fenced.iloc[counter]['gps_point_exit'])
                cmf_pos, cmf_neg = calculate_cmf(speeds_in_fence, distance)
                all_speeds.append([np.average(speeds_in_fence), cmf_pos, cmf_neg])
                speeds_in_fence = []
                counter += 1

                print('Processed row', counter, '/', total_rows)

    import pdb; pdb.set_trace()
    pd.to_csv('process_speed.csv', sep=',', index=False)


if __name__ == '__main__':
    df = pd.read_csv('Data/proov_001/speed.csv')
    df_fenced = preproc_gps_data(pd.read_csv('Data/proov_001_geoFenced.csv', header=None))

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)

    add_speeds(df, df_fenced)

    # df['date'] = df['time'].dt.date
    # df['hour'] = df['time'].dt.hour
    # df['minute'] = df['time'].dt.minute
    # df['second'] = df['time'].dt.second
    # df['minute_in_hours'] = df['minute'] * 60 + df['second']
    # df['accelerating'] = (df['speed'] - df['speed'].shift(-delta_timestep * 5)) < -delta_accel
    # df['breaking'] = (df['speed'] - df['speed'].shift(-delta_timestep * 5)) > delta_accel
    #
    # df_day_sample = df[307:75676].reset_index(drop=True)
    # df_hour_sample = df_day_sample.loc[df_day_sample['hour'] == set_hour]

    # plt.plot(df_hour_sample['minute_in_hours'], df_hour_sample['speed'])
    # plt.title("speed between %d and %d" % (set_hour, set_hour+1))
    # plt.xlabel("Time in seconds")
    # plt.ylabel("Speed in km/h")
    # plt.savefig('speed_plot.png')
    #
    # plt.show()
