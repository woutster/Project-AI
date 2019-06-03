import csv
import matplotlib.pyplot as plt
from datetime import datetime, timezone


speed = []
time = []

format = "%Y-%m-%d %H:%M:%S %Z%z"


with open('citea_e_20/speed.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            line_count += 1
            speed.append(row[1])
            time.append(datetime.utcfromtimestamp(int(row[0])/1000.))

print(time[0], time[-1])
plt.plot(speed, time)
plt.show()
