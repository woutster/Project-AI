import pandas as pd
from glob import glob
import os
import folium

from picket import Fence


def get_fence():
    geoFence = Fence()
    points = [(52.126483, 5.053564),  # Top right
              (52.116285, 5.138844),  # Top left
              (52.069820, 5.145536),  # Bottom Left
              (52.054082, 5.031153)]  # Bottom right

    for point in points:
        geoFence.add_point(point)
    return geoFence


def get_files(geo_fence, type_flag, file_flag):
    files = glob('/data/ai-projects/*')

    correct_list = []
    mean_coordinates = []
    no_coordinates = 0
    total_files = []
    unhealthy_files = []

    file_size = len(files)
    files_iterated = 0

    for file in files:

        files_iterated += 1

        exists = os.path.isfile(file + '/gps_filter.csv')
        if exists:
            filename = file.split('/')[-1]

            print('Parsing file \'', filename, '\'', files_iterated, '/', file_size)
            total_files.append(filename)
            try:
                if file_flag == 'gps':
                    df = pd.read_csv(file + '/gps_filter.csv', usecols=['gps_filter'])
                    df[['lat', 'lon']] = df['gps_filter'].str.split('|', expand=True).astype(float)
                    df = df[['lat', 'lon']]
                elif file_flag == 'location':
                    df = pd.read_csv(file + '/locations.csv', usecols=['lat', 'lon'])

            except pd.io.common.EmptyDataError:
                print('There was a problem with file', filename)
                unhealthy_files.append(filename)
                continue

            no_coordinates += df.shape[0]
            if type_flag == 'check_files':
                avg_lat, avg_lon = df.mean()
                mean_coordinates.append([filename, (avg_lat, avg_lon)])
                if geo_fence.check_point((avg_lat, avg_lon)):
                    correct_list.append(file)

            elif type_flag == 'get_points':
                raise NotImplementedError

                # for _, row in df.iterrows():
                #     point = (row['lat'], row['lon'])
                #     # if geo_fence.check_point(point):

    print('Parsed', len(total_files), 'files of which', len(unhealthy_files), 'are broken')
    print('Within the healthy files', no_coordinates, 'coordinates are parsed')
    print('')

    return correct_list, mean_coordinates, no_coordinates


def plot_map(coordinates):
    mapit = folium.Map(location=[52.0, 5.0], zoom_start=6)
    for line, coord in coordinates:
        folium.Marker(location=[coord[0], coord[1]], fill_color='#43d9de', radius=8).add_child(folium.Popup(line)).add_to(mapit)

    mapit.save('map.html')


if __name__ == '__main__':
    fence = get_fence()
    import pdb; pdb.set_trace()
    files, coords, no_coordinates = get_files(fence, type_flag='check_files', file_flag='gps')
    plot_map(coords)
