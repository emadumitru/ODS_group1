from graps import run_graps
import time

path_data = 'data/'
counties = ['RI', 'WA', 'AR']
counties_name = ['Rhode Island', 'Washington', 'Arkansas']
target_distr = [2, 10, 4]

# counties = ['AR']
# counties_name = ['Arkansas']
# target_distr = [4]

# counties = ['RI']
# counties_name = ['Rhode Island']
# target_distr = [2]

# counties = ['WA']
# counties_name = ['Washington']
# target_distr = [10]


types_districts = ['tracts']
types_districts = ['counties']
path = path_data + 'plots/'
paths = {}
for td in types_districts:
    for county, county_name, target_distr in zip(counties, counties_name, target_distr):
        graph_path = path_data + county + '/' + td + '/'
        map_path = path_data + county + '/' + td + '/' + county + '_' + td + '.shp'
        name = county_name + ' ' + td
        paths[name] = (graph_path, map_path, target_distr, county, td)

for name, (graph_path, map_path, target_distr, county, td) in paths.items():
    start_time = time.time()
    run_graps(graph_path, map_path, name, target_distr, county, path, td)
    print(f'Elapsed time for {name}: {round(time.time() - start_time, 2)}')
        