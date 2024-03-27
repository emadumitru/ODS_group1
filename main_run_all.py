from graps import run_graps

path_data = 'data/'
counties = ['RI', 'WA', 'AR']
counties_name = ['Rhode Island', 'Washington', 'Arkansas']
target_distr = [2, 10, 4]

# counties = ['AR']
# counties_name = ['Arkansas']
# target_distr = [4]

types_districts = ['counties', 'tracts']
path = path_data + 'plots/'
paths = {}
for county, county_name, target_distr in zip(counties, counties_name, target_distr):
    for td in types_districts:
        graph_path = path_data + county + '/' + td + '/'
        map_path = path_data + county + '/' + td + '/' + county + '_' + td + '.shp'
        name = county_name + ' ' + td
        paths[name] = (graph_path, map_path, target_distr, county, td)

for name, (graph_path, map_path, target_distr, county, td) in paths.items():
    run_graps(graph_path, map_path, name, target_distr, county, path, td)
        