import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_graph(path, county):

    features_df = pd.read_csv(path + county + '_features.csv', header=None, dtype={0: str}).rename(columns={0: 'GEOID', 1: 'population', 2: 'neighboors'})
    distances_df = pd.read_csv(path + county + '_distances.csv').drop(columns=['Node_ID'])

    distances_df.columns = distances_df.index

    mean_populations = round(features_df['population'].mean())
    features_df['population'] = features_df['population'].replace(0, mean_populations)


    pop_dict = features_df['population'].to_dict()
    geoid_dict = features_df['GEOID'].to_dict()
    neighboors_dict = features_df['neighboors'].to_dict()

    full_graph = nx.from_pandas_adjacency(distances_df)
    nx.set_node_attributes(full_graph, pop_dict, 'population')
    nx.set_node_attributes(full_graph, geoid_dict, 'geoid')

    G = nx.Graph()

    for node, edges in full_graph.adjacency():
        G.add_node(node, **full_graph.nodes[node])
        neighbor_geoids = neighboors_dict.get(node, []).split(',')
        
        # Iterate over the edges
        for neighbor, attributes in edges.items():
            if str(geoid_dict.get(neighbor)) in neighbor_geoids:
                G.add_edge(node, neighbor, **attributes)
    
    return G, full_graph


if __name__ == '__main__':
    test_path = 'data/RI/'

    G, full_graph = load_graph(test_path)
            
    print(nx.info(G))
    print(nx.info(full_graph))
    print(list(G.nodes(data=True))[:3])
    print(list(G.edges(data=True))[:3])

    nx.draw(G, with_labels=True)
    plt.show()