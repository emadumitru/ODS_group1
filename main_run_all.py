from algo_functions import run_algorithm
import time

path_data = 'data/'
counties = ['RI', 'WA', 'AR']
counties_name = ['Rhode Island', 'Washington', 'Arkansas']
target_distr = [2, 10, 4]

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
        