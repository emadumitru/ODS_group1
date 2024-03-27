import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_partitioned_graph(G, partitioning, name, path, name_fig, seed=70):
    """
    Plots a graph with nodes colored according to their partitioning.
    
    Parameters:
    - G: A NetworkX graph.
    - partitioning: A dictionary mapping nodes to their respective partitions.
    - name: Descriptive name for the graph partitioning.
    - path: Path where the figure will be saved.
    - name_fig: Filename for the saved figure.
    - seed: Seed for random number generator and layout reproducibility.
    """
    # Ensure reproducibility
    np.random.seed(seed)

    # Create a layout for our nodes 
    pos = nx.spring_layout(G, seed=seed)

    # Generate a color map based on the partitioning
    unique_districts = sorted(set(partitioning.values()))
    color_map = {district: plt.cm.tab10(i) for i, district in enumerate(unique_districts)}
    node_colors = [color_map[partitioning[node]] for node in G.nodes()]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the graph on the axes
    nx.draw(G, pos, ax=ax, node_color=node_colors, with_labels=True, node_size=500, edge_color='gray')

    # Create a legend for the districts
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=f'District {district}',
                          markerfacecolor=color, markersize=10) for district, color in color_map.items()]
    ax.legend(handles=patches, title='Districts', loc='best')

    # Set the title
    ax.set_title(f'Graph Partitioning: {name}', fontsize=14)

    # Save the figure
    plt.savefig(f"{path}{name_fig}", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory after saving


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

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the graph on the axes
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, edge_color='gray')

    # Set title
    ax.set_title(f'Graph: {name}', fontsize=14)
    
    # Save the figure
    plt.savefig(f"{path}{name_fig}", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory after saving

def plot_districts_from_tracts(G, partitioning, tracts_path, name, path, name_fig, district_population):
    """
    Plot tracts colored by districts based on a partitioning, including district populations in the legend.
    
    Parameters:
    - G: NetworkX graph, where each node is assumed to be an index matching tracts.
    - partitioning: Dictionary mapping node indices to district identifiers.
    - tracts_path: Path to the tracts shapefile.
    - name: Descriptive name for the plot.
    - path: Path where the figure will be saved.
    - name_fig: Name of the figure file.
    - district_population: Dictionary mapping district identifiers to their total population.
    """
    # Load the tracts shapefile
    tracts_gdf = gpd.read_file(tracts_path)
    
    # Create a DataFrame from the partitioning dictionary
    partition_df = pd.DataFrame(list(partitioning.items()), columns=['index', 'District'])
    partition_df['index'] = partition_df['index'].astype(int)
    
    # Merge the GeoDataFrame with the partition DataFrame based on index
    merged_gdf = tracts_gdf.merge(partition_df, left_index=True, right_on='index')
    
    # Generate a unique color for each district
    unique_districts = merged_gdf['District'].unique()
    colors = plt.cm.tab20(range(len(unique_districts)))  # Adjust color map as needed
    color_map = dict(zip(unique_districts, colors))
    # Apply the color map to the 'District' column to create a 'color' column
    merged_gdf['color'] = merged_gdf['District'].map(color_map)

    # Plotting with custom colors
    fig, ax = plt.subplots(figsize=(15, 15))
    for district, group in merged_gdf.groupby('District'):
        group.plot(color=color_map[district][:3], ax=ax, label=f"{district_population.get(district, 'N/A')}")

    # Custom legend for district populations
    legend_labels = [f"{population}" for district, population in district_population.items()]
    legend_handles = [mpatches.Patch(color=color_map[district][:3], label=label) for district, label in zip(unique_districts, legend_labels)]

    # Add legend to the plot inside
    plt.legend(handles=legend_handles, title="Districts", loc='upper left', prop={'size': 10})

    ax.set_aspect('equal')
    plt.title(f'Mapped districts: {name}')
    plt.savefig(f"{path}{name_fig}", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory

   