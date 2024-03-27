import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def plot_partitioned_graph(G, partitioning, name, path, name_fig, seed=70):
    """
    Plots a graph with nodes colored according to their partitioning.
    
    Parameters:
    - G: A NetworkX graph.
    - partitioning: A dictionary mapping nodes to their respective partitions.
    """
    # Ensure reproducibility
    np.random.seed(seed)

    # Create a layout for our nodes 
    pos = nx.spring_layout(G, seed=seed)

    # Generate a color map based on the partitioning
    unique_districts = sorted(set(partitioning.values()))
    color_map = {district: plt.cm.tab10(i) for i, district in enumerate(unique_districts)}
    node_colors = [color_map[partitioning[node]] for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=500, edge_color='gray')
    
    # Create a legend
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=f'District {district}',
                          markerfacecolor=color, markersize=10) for district, color in color_map.items()]
    plt.legend(handles=patches, loc='best')
    plt.title(f'Graph Partitioning: {name}')

    plt.savefig(path + name_fig)


def plot_graph(G, name, path, name_fig, seed=70):
    """
    Plots the graph without coloring nodes according to partitioning.
    
    Parameters:
    - G: A NetworkX graph.
    """
    # Ensure reproducibility
    np.random.seed(seed)

    # Create a layout for our nodes 
    pos = nx.spring_layout(G, seed=seed)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, edge_color='gray')
    plt.title(f'Graph: {name}')
    
    plt.savefig(path + name_fig)

def plot_districts_from_tracts(G, partitioning, tracts_path, name, path, name_fig):
    """
    Plot tracts colored by districts based on a partitioning.
    
    Parameters:
    - G: NetworkX graph, where each node is assumed to be an index matching tracts.
    - partitioning: Dictionary mapping node indices to district identifiers.
    - tracts_path: Path to the tracts shapefile.
    """
    # Load the tracts shapefile
    tracts_gdf = gpd.read_file(tracts_path)
    
    # Create a DataFrame from the partitioning dictionary
    partition_df = pd.DataFrame(list(partitioning.items()), columns=['index', 'District'])
    partition_df['index'] = partition_df['index'].astype(int)
    
    # Merge the GeoDataFrame with the partition DataFrame based on index
    # Assumes tracts_gdf has a corresponding index to link with partition_df
    merged_gdf = tracts_gdf.merge(partition_df, left_index=True, right_on='index')
    
    # Setup the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot the merged GeoDataFrame, coloring by district
    merged_gdf.plot(column='District', ax=ax, legend=True, categorical=True, legend_kwds={'bbox_to_anchor': (1, 1)})
    
    # Adjust the plot
    ax.set_aspect('equal')
    # plt.axis('on')
    plt.title(f'Tracts by District: {name}')
    plt.savefig(path + name_fig)

   