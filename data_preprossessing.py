import csv
import pandas as pd
import datetime
from datetime import datetime
import numpy as np

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
    df['time'] = df['in_time'].dt.round("S")

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

    df = pd.read_csv(filename, names=['time', 'temp', 'precipitation_intensity', 'windSpeed', 'visibility'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df['time'] = df['time'].dt.round("S")
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['second'] = df['time'].dt.second
    df['minute_in_hours'] = df['minute'] * 60 + df['second']
    return df

def match_weather_with_geo(geo, weather):
    # This is not working at all. :( 
    # print(weather['time'])
    # geo['eh'] = np.where((geo['time'] == weather['time']) | (geo['out_time'] == weather['time']), weather['temp'], np.nan)
    df = geo.merge(weather, how='left')
    print(df['temp'])

def pd_to_csv(geo, one_hot_days, weather):
    pass

geo, one_hot_days = csv_to_pd('Data/proov_001/proov_001_geoFenced.csv')
weather = weather_csv_to_pd('Data/proov_001/weather.csv')

match_weather_with_geo(geo, weather)
