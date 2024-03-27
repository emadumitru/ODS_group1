import pandas as pd
import networkx as nx
import numpy as np
from visuals import *
from loading import load_graph


def create_subgraphs_from_partitioning(G, partitioning):
    """
    Creates subgraphs based on a given partitioning of the nodes.

    Parameters:
    - G: A NetworkX graph.
    - partitioning: A dictionary with node IDs as keys and district IDs as values.

    Returns:
    - A dictionary of subgraphs, with district IDs as keys.
    """
    subgraphs = {}
    # Initialize a list to store nodes for each district
    districts = {}
    for node, district in partitioning.items():
        if district not in districts:
            districts[district] = []
        districts[district].append(node)
    
    # Create a subgraph for each district
    for district, nodes in districts.items():
        subgraphs[district] = G.subgraph(nodes).copy()
    
    return subgraphs

def calculate_district_populations(G, partitioning):
    """
    Calculates the total population for each district given a graph and a partitioning.

    Parameters:
    - G: A NetworkX graph, where each node has a 'population' attribute.
    - partitioning: A dictionary with node IDs as keys and district IDs as values.

    Returns:
    - A dictionary with district IDs as keys and the total population of each district as values.
    """
    district_populations = {}
    for node, district in partitioning.items():
        if district not in district_populations:
            district_populations[district] = 0
        district_populations[district] += G.nodes[node]['population']
    
    return district_populations

def calculate_rmse_and_top_district(district_populations, target_population):
    """
    Calculates the mean RMSE between the district populations and a target population,
    and identifies the district with the highest population.

    Parameters:
    - district_populations: A dictionary with district IDs as keys and populations as values.
    - target_population: The target population for each district.

    Returns:
    - mean_rmse: The mean RMSE between the district populations and the target population.
    - top_district: The ID of the district with the highest population.
    """
    # Calculate squared differences
    squared_diffs = [(pop - target_population) ** 2 for pop in district_populations.values()]
    
    # Calculate mean RMSE
    mean_rmse = np.sqrt(np.mean(squared_diffs))
    
    # Identify the district with the highest population
    top_district = max(district_populations, key=district_populations.get)
    
    return mean_rmse, top_district

def check_subgraphs_connectivity(district_subgraphs):
    """
    Checks if all nodes in each subgraph are connected using only paths within the subgraph.

    Parameters:
    - district_subgraphs: A dictionary with district identifiers as keys and the corresponding
      NetworkX subgraphs as values.

    Returns:
    - A boolean indicating whether all subgraphs are connected (True) or if there's at least
      one subgraph that is not connected (False).
    """
    for district, subgraph in district_subgraphs.items():
        if not nx.is_connected(subgraph):
            # If the subgraph is not connected, return False
            return False
    # If all subgraphs are connected, return True
    return True

def calculate_total_distance(G, nodes):
    """
    Calculates the total distance within a subgraph of G defined by `nodes`,
    as the sum of all shortest path lengths between pairs of nodes.
    """
    subgraph = G.subgraph(nodes)
    total_distance = sum(nx.single_source_shortest_path_length(subgraph, node).values() for node in subgraph.nodes())
    # Since each path is counted twice (once from each node), divide by 2
    return total_distance / 2

def calculate_distance_impact(G, partitioning, node, current_district, new_district):
    """
    Calculates the distance impact of moving a node from its current district to a new district.
    The impact is defined as the change in the total distance within the affected districts.
    
    Parameters:
    - G: A NetworkX graph.
    - partitioning: Current partitioning as a dictionary {node: district}.
    - node: The node to be moved.
    - current_district: The current district of the node.
    - new_district: The district to move the node to.
    
    Returns:
    - The distance impact of the move.
    """
    # Get the nodes for each district before the move
    current_district_nodes = [n for n, d in partitioning.items() if d == current_district and n != node]
    new_district_nodes = [n for n, d in partitioning.items() if d == new_district] + [node]

    # Calculate total distance before the move
    before_move_distance = calculate_total_distance(G, current_district_nodes + [node]) + calculate_total_distance(G, new_district_nodes[:-1])

    # Calculate total distance after the move
    after_move_distance = calculate_total_distance(G, current_district_nodes) + calculate_total_distance(G, new_district_nodes)

    # The impact is the difference in total distance
    return after_move_distance - before_move_distance

def heuristic_node_selection(G, partitioning, district_largest_population, nonvalid_partitions):
    """
    Selects a node from the district with the largest total population to move to another district,
    based on improvement in RMSE, internal district distances, and partition validity.

    Parameters:
    - G: A NetworkX graph.
    - partitioning: A dictionary mapping nodes to their current district.
    - district_largest_population: The district with the largest total population.
    - nonvalid_partitions: A list to keep track of non-valid partitions for optimization.

    Returns:
    - The node to move and the district to move it to.
    - Updated list of non-valid partitions.
    """
    best_move = (None, None)  # (node to move, district to move to)
    best_improvement = np.inf  # Track the improvement; lower RMSE is better
    best_distance_impact = np.inf  # Placeholder for evaluating distance impact

    # Calculate current RMSE
    current_rmse, _ = calculate_rmse_and_top_district(calculate_district_populations(G, partitioning), np.mean(list(calculate_district_populations(G, partitioning).values())))
    
    for node in partitioning:
        if partitioning[node] == district_largest_population:
            # Simulate moving the node to each possible district
            for potential_district in set(partitioning.values()):
                if potential_district != district_largest_population:
                    # Simulate the move
                    new_partitioning = partitioning.copy()
                    new_partitioning[node] = potential_district
                    
                    # Skip calculation if this partitioning is already known to be non-valid
                    if tuple(sorted(new_partitioning.items())) in nonvalid_partitions:
                        continue
                    
                    # Check if the new partitioning is valid (all subgraphs are connected)
                    subgraphs = create_subgraphs_from_partitioning(G, new_partitioning)
                    if not check_subgraphs_connectivity(subgraphs):
                        nonvalid_partitions.append(tuple(sorted(new_partitioning.items())))
                        continue  # Skip to the next potential move
                    
                    # Calculate new RMSE after the move
                    new_rmse, _ = calculate_rmse_and_top_district(calculate_district_populations(G, new_partitioning), np.mean(list(calculate_district_populations(G, new_partitioning).values())))
                    
                    # Calculate the distance impact of the move
                    distance_impact = calculate_distance_impact(G, new_partitioning, node, district_largest_population, potential_district)
                    
                    # Evaluate the move based on RMSE improvement and distance impact
                    if new_rmse < best_improvement and distance_impact < best_distance_impact:
                        best_improvement = new_rmse
                        best_distance_impact = distance_impact
                        best_move = (node, potential_district)
    
    return best_move, nonvalid_partitions


def graph_partitioning(G, num_partitions, seed=70):
    """
    Partitions a graph into a given number of districts, optimizing for population balance and connectivity,
    ensuring that all final partitions are valid (i.e., all subgraphs are connected).

    Parameters:
    - G: A NetworkX graph, where each node has a 'population' attribute.
    - num_partitions: The desired number of partitions (districts).

    Returns:
    - The final partitioning of nodes into districts.
    """
    # Initial setup
    np.random.seed(seed)  # Set the seed for reproducibility
    nodes = sorted(G.nodes())
    np.random.shuffle(nodes)  # Randomly shuffle nodes for initial assignment
    initial_partitioning = {node: i % num_partitions for i, node in enumerate(nodes)}
    nonvalid_partitions = [tuple(sorted(initial_partitioning.items()))]  # Include initial partitioning as non-valid to force re-calculation

    # Start with an empty partitioning to build upon
    partitioning = {}

    while True:
        # Attempt to construct a valid partitioning from scratch or from the partial valid base
        for node in nodes:
            for district in range(num_partitions):
                # Temporarily assign or reassign the node to a district
                temp_partitioning = partitioning.copy()
                temp_partitioning[node] = district
                
                # Skip if we've already determined this partitioning is non-valid
                if tuple(sorted(temp_partitioning.items())) in nonvalid_partitions:
                    continue

                # Check if current temp partitioning is valid
                subgraphs = create_subgraphs_from_partitioning(G, temp_partitioning)
                if check_subgraphs_connectivity(subgraphs):
                    partitioning = temp_partitioning  # Accept the change
                    break  # Break to try next node with this updated partitioning
            else:
                # If no valid district was found for the current node, mark the partition as non-valid
                nonvalid_partitions.append(tuple(sorted(temp_partitioning.items())))
                continue  # Continue trying with the next node
                
            # If we've successfully assigned all nodes with valid connectivity
            if len(partitioning) == len(nodes):
                return partitioning
            

def partition_subgraphs_and_allocate(G, num_partitions):
    """
    Identifies connected subgraphs, prints their sizes, and allocates a specified number of
    partitions across these subgraphs based on their sizes. Returns the largest subgraph
    and a dictionary that maps subgraphs to the number of allocated partitions.

    Parameters:
    - G: A NetworkX graph.
    - num_partitions: The desired number of partitions to allocate across subgraphs.

    Returns:
    - largest_subgraph: The largest subgraph in terms of number of nodes.
    - partition_allocation: A dictionary mapping each subgraph to the number of allocated partitions.
    """
    # Identify all connected components (subgraphs)
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    subgraph_sizes = [len(subgraph.nodes()) for subgraph in subgraphs]
    
    # Print the number of subgraphs and their sizes
    print(f"Number of connected subgraphs: {len(subgraphs)}")
    print("Sizes of subgraphs:", subgraph_sizes)
    
    # Find the largest subgraph
    largest_subgraph = max(subgraphs, key=lambda sg: len(sg.nodes()))
    
    partition_allocation = {}
    if len(subgraphs) > num_partitions:
        print("More non-connected subgraphs than districts.")
        for subgraph in subgraphs:
            partition_allocation[subgraph] = 0
    else:
        total_nodes = sum(subgraph_sizes)
        remaining_partitions = num_partitions

        # Allocate partitions based on sizes, ensuring at least 1 partition per subgraph
        for i, (subgraph, size) in enumerate(zip(subgraphs, subgraph_sizes)):
            if i < len(subgraphs) - 1:  # Not the last subgraph
                partitions = max(1, round((size / total_nodes) * num_partitions))
                partitions = min(partitions, remaining_partitions - (len(subgraphs) - i - 1))  # Ensure remaining subgraphs get at least 1
            else:  # For the last subgraph, allocate all remaining partitions
                partitions = remaining_partitions
            
            partition_allocation[subgraph] = partitions
            remaining_partitions -= partitions
            
    return largest_subgraph, partition_allocation

def comprehensive_partitioning(G, num_partitions):
    """
    Adjusts the partitioning strategy to focus on the largest subgraph first, then aggregates
    additional partitioning from other subgraphs if applicable.

    Parameters:
    - G: A NetworkX graph.
    - num_partitions: The total number of partitions desired.

    Returns:
    - The partitioning of the largest subgraph.
    - The aggregated partitioning from other subgraphs, or None if not applicable.
    """
    # Step 1: Identify subgraphs and allocate partitions
    largest_subgraph, partition_allocation = partition_subgraphs_and_allocate(G, num_partitions)

    # Partition the largest subgraph if more than one partition is allocated
    largest_subgraph_partitioning = graph_partitioning(largest_subgraph, num_partitions)

    # Determine if there are other subgraphs to process
    if len(partition_allocation) > 1:
        final_partitioning = {}
        idx = 0
        
        for subgraph, partitions in partition_allocation.items():
            if partitions > 1:
                partition_sub = graph_partitioning(subgraph, partitions)
                part_id = [i + idx for i in range(partitions)]
                remap_dict = dict(zip(range(partitions), part_id))
                print(remap_dict)
                final_partitioning.update({k: remap_dict[v] for k, v in partition_sub.items()})
                idx += partitions
            else:
                final_partitioning.update({k: idx for k in subgraph.nodes()})
                idx += 1

    else:
        # If there's only the largest subgraph, no additional partitioning to aggregate
        final_partitioning = None

    return largest_subgraph_partitioning, final_partitioning, largest_subgraph


def run_graps(data_graph, data_map, name, target_part, county, path, type_dist):
    G, full = load_graph(data_graph, county)
    county = county + '_' + type_dist
    plot_graph(G, name, path, county + '_graph.png')
    partitions = comprehensive_partitioning(G, target_part)
    if partitions[1] is not None:
        plot_partitioned_graph(partitions[2], partitions[0], name + ' biggest subgraph', path, county + '_biggest_subgraph.png')
        plot_districts_from_tracts(partitions[2], partitions[0], data_map, name + ' biggest subgraph', path, county + '_biggest_subgraph_map.png')
        plot_partitioned_graph(G, partitions[1], name + ' multipe subgraphs', path, county + '_multipe_subgraphs.png')
        plot_districts_from_tracts(G, partitions[1], data_map, name + ' multipe subgraphs', path, county + '_multipe_subgraphs_map.png')
    else:
        plot_partitioned_graph(G, partitions[0], name, path, county + '_single_subgraph.png')
        plot_districts_from_tracts(G, partitions[0], data_map, name, path, county + '_single_subgraph_map.png')