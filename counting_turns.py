import numpy as np
import pandas as pd
import rdp
from geopy import distance

def angle(gps, index):
    r = [np.NaN]
    for i in range(0, gps.shape[0]-2):
        A = gps.iloc[i]
        B = gps.iloc[i+1]
        C = gps.iloc[i+2]

        a = distance_between(C, B)
        b = distance_between(A, C)
        c = distance_between(A, B)

        if c < 5 or a < 5:
            r.append(np.NaN)
            continue

        theta = np.arccos((b**2 - a**2 - c**2) / 2 / a / c)

        r.append(theta*180/np.pi)
    r.append(np.NaN)
    return pd.Series(r, name='angle', index=index)


def distance_between(point1, point2):
    return distance.distance((point1['lat'], point1['lon']), (point2['lat'], point2['lon'])).meters

def n_turns(df):
    print("in turns")
    mask = rdp.rdp(df[['lat', 'lon']].values,
                   epsilon=5e-5,
                   return_mask=True,
    )
    print("masked")
    track = df[mask].copy()
    track['angle'] = angle(track, track.index)
    track.dropna(inplace=True)
    return (track["angle"] > 20).sum()

#
# df = pd.read_csv(f'Data/proov_001/gps_filter.csv')
# # print(df.columns)
# new = df['gps_filter'].str.split('|', n = 1, expand=True)
# df['lat'] = new[0].astype(float)
# df['lon'] = new[1].astype(float)
# print(df.columns)
#
# print(n_turns(df))
