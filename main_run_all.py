from algo_functions import run_algorithm
import time

path_data = 'data/'
counties = ['RI', 'WA', 'AR']
counties_name = ['Rhode Island', 'Washington', 'Arkansas']
target_distr = [2, 10, 4]

# longer_version
counties_name = ["Alabama", "Alaska", "Arizona", "Arkansas", "Connecticut", 
                 "Georgia", "Idaho", "Illinois", "Indiana", "Iowa", 
                 "North Dakota", "South Dakota", "Colorado", "Kansas", "Virginia", 
                 "Oklahoma", "Rhode Island", "Washington", "Mississippi", "New Mexico"]
counties = ["AL", "AK", "AZ", "AR", "CT", 
            "GA", "ID", "IL", "IN", "IA", 
            "ND", "SD", "CO", "KS", "VT", 
            "OK", "RI", "WA", "MS", "NM"]
target_distr = [7, 1, 9, 4, 5, 
                14, 2, 17, 9, 4, 
                1, 1, 8, 4, 11, 
                5, 2, 10, 4, 3]

path = 'plots/'
paths = {}
for county, county_name, target_distr in zip(counties, counties_name, target_distr):
    graph_path = path_data
    map_path = path_data + county + '_' + 'counties' + '.shp'
    name = county_name
    paths[name] = (graph_path, map_path, target_distr, county)

for name, (graph_path, map_path, target_distr, county) in paths.items():
    start_time = time.time()
    run_algorithm(graph_path, map_path, name, target_distr, county, path, 'counties')
    print(f'Elapsed time for {name}: {round(time.time() - start_time, 2)}')
        