import csv
import pandas as pd
import datetime
from datetime import datetime
import numpy as np
import requests


import ast
import geopy.distance
import matplotlib.pyplot as plt

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
    # df['hour'] = df['time'].dt.hour
    # df['minute'] = df['time'].dt.minute
    # df['second'] = df['time'].dt.second
    # df['minute_in_hours'] = df['minute'] * 60 + df['second']
    return df

def match_weather_with_geo(geo, weather):
    # This is not working at all. :(
    df = pd.merge(geo, weather, left_on='date', right_on='date')
    # df = df.drop_duplicates(subset='in_time', keep='first')
    del df['in']
    del df['out']
    print(df.columns)
    geo['temp'] = 0

    # print('doing stuff...')
    # print(geo['temp'])

def pd_to_csv(filename, geo, weather, one_hot_days):

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

    holiday_boolean_list = []
    for i, g in geo.iterrows():
        if str(g['date']) in holidates:
            holiday_boolean_list.append(1)
        else:
            holiday_boolean_list.append(0)


    geo['holiday'] = holiday_boolean_list

    return geo


def add_speed(filename, geo):
    speed = {'time': [], 'speed': []}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue
                speed['time'].append(float(row[0]))
                speed['speed'].append(float(row[1]))

        except:
            pass

    df = pd.DataFrame.from_dict(speed)

    # Fix time format
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['time'] = df['time'].dt.round("S")
    df['avg_speed'] = df['speed'].copy()

    cmf_poss, cmf_negs, avg_speed, speeds_in_fence, all_speeds, distances = [], [], [], [], [], []

    for i, g in geo.iterrows():
        if i < 10:
            continue
        for j, s in df.iterrows():
            # all_speeds.append(s['speed'])
            print('In:', g['in_time'], s['time'], 'out:', g['out_time'])
            # if g['out_time'] < s['time']:
            #     break
            if g['in_time'] >= s['time']:
                # print('In:', g['in_time'], s['time'])
                speeds_in_fence.append(s['speed'])
                if g['out_time'] <= s['time']:
                    print('out')
                    # print('Out:', g['out_time'], s['time'])
                    distance = calculate_distance(f"({g['in_lat']}, {g['in_long']})", f"({g['out_lat']}, {g['out_long']})")
                    cmf_pos, cmf_neg = calculate_cmf(speeds_in_fence, distance)
                    # all_speeds.append([np.average(speeds_in_fence), cmf_pos, cmf_neg])

                    distances.append(distance)
                    cmf_poss.append(cmf_pos)
                    cmf_negs.append(cmf_neg)
                    avg_speed.append(np.average(speeds_in_fence))

                    speeds_in_fence = []

        print(len(cmf_poss))

    geo['distance'] = distances
    geo['cmf_pos'] = cmf_poss
    geo['cmf_neg'] = cmf_neg

    return geo

bus = 'proov_002'
geo, one_hot_days = csv_to_pd(f'Data/{bus}/{bus}_geoFenced.csv')
weather = weather_csv_to_pd(f'Data/{bus}/weather.csv')

geo = add_speed("Data/proov_001/speed.csv", geo)
# match_weather_with_geo(geo, weather)
# geo = holidays(geo)
# pd_to_csv(f'Data/{bus}/combined.csv', geo, weather, one_hot_days)
