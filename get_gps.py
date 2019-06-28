import pandas as pd
from glob import glob
import os
import folium

from picket import Fence


def get_fence(points):
    """ Initialises geofence. """
    geoFence = Fence()
    for point in points:
        geoFence.add_point(point)
    return geoFence


def get_files(geo_fence, file_flag, type_flag):
    """ Get necessary files.
    Please note that directory has to be changed accordingly.
    """
    files = glob('/data/ai-projects/*')

    files_to_consider = ['proov_001', 'proov_002', 'proov_003']

    correct_list = []
    mean_coordinates = []
    no_coordinates = 0
    total_files = []
    unhealthy_files = []

    file_size = len(files)
    files_iterated = 0

    geo_fence_list = []

    # Iterate over files
    for file in files:

        files_iterated += 1

        # Ensure file exits
        exists = os.path.isfile(file + '/' + file_flag + '.csv')
        if exists:
            filename = file.split('/')[-1]

            # Filter unnecessary files
            if filename in files_to_consider:

                print('Parsing file \'', filename, '\'', files_iterated, '/', file_size)
                total_files.append(filename)
                try:
                    # If the file is of all the gps coordinates
                    # Take the coordinates and put them in a dataframe
                    if file_flag == 'gps_filter':
                        df = pd.read_csv(file + '/gps_filter.csv')
                        df[['lat', 'lon']] = df['gps_filter'].str.split('|', expand=True).astype(float)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.drop('gps_filter', axis=1, inplace=True)
                    # If the file is of the locations (sparser gps data)
                    # Read it into a dataframe
                    elif file_flag == 'locations':
                        df = pd.read_csv(file + '/locations.csv', usecols=['lat', 'lon'])

                except pd.io.common.EmptyDataError:
                    print('There was a problem with file', filename)
                    unhealthy_files.append(filename)
                    continue

                no_coordinates += df.shape[0]
                # Just check which buses drive to the geofence
                if type_flag == 'check_files':
                    avg_lat, avg_lon = df.mean()
                    mean_coordinates.append([filename, (avg_lat, avg_lon)])
                    if geo_fence.check_point((avg_lat, avg_lon)):
                        correct_list.append(file)

                # Get all the points of the buses that drive trough the geofence
                elif type_flag == 'get_points':

                    pass_counter = 0
                    pass_trough_bool = False
                    pass_trough_time_in = []
                    pass_trough_list = []

                    for _, row in df.iterrows():
                        point = (row['lat'], row['lon'])
                        if geo_fence.check_point(point):
                            pass_counter += 1
                            if not pass_trough_bool:
                                pass_trough_time_in = [point, row['time']]
                            pass_trough_bool = True
                            geo_fence_list.append(point)
                        else:
                            if pass_trough_bool:
                                pass_trough_bool = False
                                pass_trough_list.append([pass_trough_time_in, [point, row['time']]])

                    df_troughput = pd.DataFrame(pass_trough_list)
                    df_troughput.to_csv(filename + '_geoFenced.csv', header=False, index=False)

    print('Parsed', len(files), 'files of which', len(files) - len(total_files), 'are not availlable to parse')
    print('Within the healthy files', no_coordinates, 'coordinates are parsed')
    print('')

    return correct_list, mean_coordinates, no_coordinates


def plot_map(coordinates, file_flag, points):
    """Plot a map that shows the coordinates"""
    mapit = folium.Map(location=[52.0, 5.0], zoom_start=6)
    folium.vector_layers.Polygon(points).add_to(mapit)
    for line, coord in coordinates:
        folium.Marker(location=[coord[0], coord[1]], fill_color='#43d9de', radius=8).add_child(folium.Popup(line)).add_to(mapit)

    mapit.save('map_' + file_flag + '.html')


def run_get_gps():
    file_flag = 'gps_filter'
    type_flag = 'get_points'
    points = [(52.093338, 5.111842),  # Top left
              (52.093549, 5.116343),  # Top right
              (52.092675, 5.116692),  # Bottom Left
              (52.092632, 5.112223)]  # Bottom right
    fence = get_fence(points)
    files, coords, no_coordinates = get_files(fence, file_flag, type_flag)
    # plot_map(coords, file_flag, points)


if __name__ == '__main__':
    file_flag = 'gps_filter'
    type_flag = 'get_points'
    points = [(52.093338, 5.111842),  # Top left
              (52.093549, 5.116343),  # Top right
              (52.092675, 5.116692),  # Bottom Left
              (52.092632, 5.112223)]  # Bottom right
    fence = get_fence(points)
    files, coords, no_coordinates = get_files(fence, file_flag, type_flag)
    # plot_map(coords, file_flag, points)
