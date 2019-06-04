import pandas as pd
import matplotlib.pyplot as plt

start_date = '2017-01-02'
end_date = '2017-01-02'
set_hour = 7

# Timestep to calculate if vehicle is accelerating/braking in seconds
delta_accel = 4
delta_timestep = 1


df = pd.read_csv('Data/proov_001/speed.csv')
df['time'] = pd.to_datetime(df['time'], unit='ms')
df['date'] = df['time'].dt.date
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['second'] = df['time'].dt.second
df = df.drop('time', axis=1)
df['minute_in_hours'] = df['minute'] * 60 + df['second']
df['accelerating'] = (df['speed'] - df['speed'].shift(-delta_timestep * 5)) < -delta_accel
df['breaking'] = (df['speed'] - df['speed'].shift(-delta_timestep * 5)) > delta_accel

df_day_sample = df[307:75676].reset_index(drop=True)
df_hour_sample = df_day_sample.loc[df_day_sample['hour'] == set_hour]

# import pdb; pdb.set_trace()

plt.plot(df_hour_sample['minute_in_hours'], df_hour_sample['speed'])
plt.title("speed between %d and %d" % (set_hour, set_hour+1))
plt.xlabel("Time in seconds")
plt.ylabel("Speed in km/h")
plt.savefig('speed_plot.png')

plt.show()
