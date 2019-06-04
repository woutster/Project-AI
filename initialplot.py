import csv
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import numpy as np
from scipy.signal import savgol_filter


speed16 = []
speed19 = []
time16 = []
time19 = []


with open('../Data/proov_001/speed.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    init_day = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        elif line_count == 1:
            init_day = datetime.utcfromtimestamp(int(row[0])/1000.)
            init_day = init_day.day
            print(datetime.utcfromtimestamp(int(row[0])/1000.))
        line_count += 1
        dateandtime = datetime.utcfromtimestamp(int(row[0])/1000.)
        # print(dateandtime)
        if dateandtime.day > 5:
            break
        if dateandtime.hour < 16:
            continue
        elif dateandtime.hour == 21 and dateandtime.day == 5:
            speed16.append(np.round(float(row[1]),2))
            time16.append(dateandtime.minute)
        elif dateandtime.hour == 17:
            speed19.append(np.round(float(row[1]),2))
            time19.append(dateandtime.minute)



# print(time[/], time[-1])

# smoothed = savgol_filter(speed, 1001, 3)
# plt.plot(speed)
plt.plot(time16, speed16)
plt.title("speed between 21:00 and 22:00")
plt.savefig('speed21.png')

plt.show()
