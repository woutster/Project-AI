import pandas as pd
from picket import Fence

geoFence = Fence()
points = [(52.0874328613281, 5.10679960250855),
          (52.0881729125977, 5.11815786361694),
          (52.0887107849121, 5.11708736419678)]

for point in points:
    geoFence.add_point(point)

files = ['Data/proov_001/locations.csv']

for file in files:
    df = pd.read_csv(file, usecols=['lat', 'lon'])
    for _, row in df.iterrows():
        point = (row['lat'], row['lon'])
        if geoFence.check_point(point):

