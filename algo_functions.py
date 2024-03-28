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

def check_subgraphs_connectivity(subgraphs):
    """
    Checks if all nodes in each provided subgraph are connected.

    Parameters:
    - subgraphs: A dictionary of subgraphs to check for connectivity.

    Returns:
    - True if all subgraphs are connected; False otherwise.
    """
    for subgraph in subgraphs.values():
        if not nx.is_connected(subgraph):
            return False
    return True

def heuristic_node_selection(G, partitioning, nonvalid_partitions, previous_states):
    """
    Selects a node to move from one district to another to improve overall population balance,
    while ensuring connectivity within districts is maintained.

    Parameters:
    - G: A NetworkX graph.
    - partitioning: Current partitioning as a dictionary {node: district}.
    - district_populations: Current populations for each district as a dictionary {district: population}.
    - nonvalid_partitions: A list of partitionings known to be invalid, for optimization.
    - previous_states: A list of previously encountered partition states to avoid repeats.

    Returns:
    - best_move: A tuple (node, district) representing the best node to move and its new district.
    - nonvalid_partitions: Updated list of non-valid partitions.
    - previous_states: Updated list of previous states.
    """
    best_move = (None, None)
    populations = calculate_district_populations(G, partitioning)
    lowest_population_variance = np.var(list(populations.values()))

    for node, current_district in partitioning.items():
        for potential_district in set(partitioning.values()):
            if potential_district == current_district:
                continue

            new_partitioning = partitioning.copy()
            new_partitioning[node] = potential_district

            if tuple(sorted(new_partitioning.items())) in previous_states or tuple(sorted(new_partitioning.items())) in nonvalid_partitions:
                continue

            # Simulate the population change
            new_district_populations = populations.copy()
            node_population = G.nodes[node]['population']
            new_district_populations[current_district] -= node_population
            new_district_populations[potential_district] += node_population

            new_variance = np.var(list(new_district_populations.values()))

            subgraphs = create_subgraphs_from_partitioning(G, new_partitioning)
            if not check_subgraphs_connectivity(subgraphs) or len(subgraphs) != len(set(partitioning.values())):
                nonvalid_partitions.append(tuple(sorted(new_partitioning.items())))
                continue

            if new_variance < lowest_population_variance:
                lowest_population_variance = new_variance
                best_move = (node, potential_district)

    if best_move[0] is not None:
        new_partitioning = partitioning.copy()
        new_partitioning[best_move[0]] = best_move[1]
        previous_states.append(tuple(sorted(new_partitioning.items())))

    return best_move, nonvalid_partitions, previous_states

def create_initial_partition(G, num_partitions):    
    # Sort nodes by population
    sorted_nodes_by_population = sorted(G.nodes(data=True), key=lambda x: x[1]['population'], reverse=False)
    
    # Initial partitioning: most connected nodes are seeded into separate partitions, others start in partition 0
    partitioning = {node: 0 for node in G.nodes()}  # Start with all nodes in partition 0
    
    # Assign each of the top connected nodes to a unique partition, ensuring each partition has a highly connected node
    for i in range(1, min(num_partitions, len(sorted_nodes_by_population))):  # Start from 1 since partition 0 is the default
        partitioning[sorted_nodes_by_population[i - 1][0]] = i

    return partitioning

def graph_partitioning(G, num_partitions, seed=70):
    np.random.seed(seed)  # Ensure reproducibility
    nodes = sorted(G.nodes())
    
    # Initial partitioning: distribute nodes evenly to ensure the expected number of districts
    partitioning = create_initial_partition(G, num_partitions)
    nonvalid_partitions = []  # Start with an empty list; it will be filled with attempted non-valid partitionings
    previous_states = []

    for _ in range(1000):  # Limit the number of iterations to avoid infinite loops
        best_move, nonvalid_partitions, previous_states = heuristic_node_selection(G, partitioning, nonvalid_partitions, previous_states)
        
        if best_move[0] is None:
            # If no valid move was found, stop the iteration
            break

        # Temporarily execute the best move to test connectivity
        temp_partitioning = partitioning.copy()
        temp_partitioning[best_move[0]] = best_move[1]

        # Check connectivity after the move
        subgraphs = create_subgraphs_from_partitioning(G, temp_partitioning)
        if check_subgraphs_connectivity(subgraphs):
            # If the move maintains connectivity, make it permanent
            partitioning = temp_partitioning
        else:
            # If connectivity is lost, add to nonvalid partitions and look for another move
            nonvalid_partitions.append(tuple(sorted(temp_partitioning.items())))
            continue  # Proceed to the next iteration without updating the partitioning

    # Final check to ensure we have the desired number of partitions
    final_districts = set(partitioning.values())
    if len(final_districts) != num_partitions:
        print("Adjustments required: The final partitioning does not have the expected number of districts.")

    print(partitioning)

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
                final_partitioning.update({k: remap_dict[v] for k, v in partition_sub.items()})
                idx += partitions
            else:
                final_partitioning.update({k: idx for k in subgraph.nodes()})
                idx += 1

    else:
        # If there's only the largest subgraph, no additional partitioning to aggregate
        final_partitioning = None

    return largest_subgraph_partitioning, final_partitioning, largest_subgraph


def run_algorithm(data_graph, data_map, name, target_part, county, path, type_dist):
    G, _ = load_graph(data_graph, county)
    plot_graph(G, name, path, county + '_graph.png')
    partitions = comprehensive_partitioning(G, target_part)
    if partitions[1] is not None:
        district_population_b = calculate_district_populations(G, partitions[0])
        district_population_m = calculate_district_populations(G, partitions[1])
        print(f'Population for {name} biggest subgraph: {district_population_b}')
        print(f'Population for {name} multiple subgraphs: {district_population_m}')
        plot_partitioned_graph(partitions[2], partitions[0], name + ' biggest subgraph', path, county + '_biggest_subgraph.png')
        plot_districts_from_tracts(partitions[2], partitions[0], data_map, name + ' biggest subgraph', path, county + '_biggest_subgraph_map.png', district_population_b)
        plot_partitioned_graph(G, partitions[1], name + ' multipe subgraphs', path, county + '_multipe_subgraphs.png')
        plot_districts_from_tracts(G, partitions[1], data_map, name + ' multipe subgraphs', path, county + '_multipe_subgraphs_map.png', district_population_m)
    else:
        district_population_s = calculate_district_populations(G, partitions[0])
        print(f'Population for {name} single subgraph: {district_population_s}')
        plot_partitioned_graph(G, partitions[0], name, path, county + '_single_subgraph.png')
        plot_districts_from_tracts(G, partitions[0], data_map, name, path, county + '_single_subgraph_map.png', district_population_s)