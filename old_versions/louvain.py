from loading import load_graph
import networkx as nx
from community import community_louvain
import numpy as np
import random
random.seed(42)
np.random.seed(42)

G = load_graph('data/RI/')

def get_total_population(G):
    """
    Get the total population of the graph.
    """
    return sum(nx.get_node_attributes(G, 'population').values())

def evaluate_districts(graph, partition, pop_attr='population'):
    """
    Evaluates the new metric for given districts.
    - graph: The networkx graph
    - partition: A dict mapping node to district
    - pop_attr: The attribute name for population
    Returns: tuple of (population variance, total intra-district distance)
    """
    # Calculate population per district
    district_pops = {}
    for node, district in partition.items():
        pop = graph.nodes[node].get(pop_attr, 0)
        if district in district_pops:
            district_pops[district] += pop
        else:
            district_pops[district] = pop
            
    # Calculate population variance
    pop_variance = np.var(list(district_pops.values()))
    
    # Calculate total intra-district distance per district
    distric_dist = {}
    for u, v, data in graph.edges(data=True):
        if partition[u] == partition[v]:  # Edge within the same district
            district = partition[u]
            if district in distric_dist:
                distric_dist[district] += data.get('weight', 1)
            else:
                distric_dist[district] = data.get('weight', 1)
            
    return pop_variance, distric_dist, district_pops

def calculate_population_variance(graph, partition):
    """
    Calculate the variance of the population across districts in the partition.
    """
    district_populations = {}
    for node, district in partition.items():
        pop = graph.nodes[node].get('population', 0)
        if district not in district_populations:
            district_populations[district] = 0
        district_populations[district] += pop
    
    populations = list(district_populations.values())
    if not populations:  # Prevent division by zero
        return 0
    return np.var(populations)

def calculate_intra_district_distance(graph, partition):
    """
    Calculate the total distance within districts, encouraging compactness.
    """
    total_distance = 0
    for (u, v, data) in graph.edges(data=True):
        if partition[u] == partition[v]:  # Nodes belong to the same district
            total_distance += data.get('weight', 1)  # Assuming 'weight' represents distance
    return total_distance

def evaluation(graph, partition, target_partitions):
    """
    Custom metric considering population distribution and internal/external mean distances.
    
    Parameters:
    - graph: The graph object.
    - partition: A dictionary mapping each node to its district.
    - target_partitions: The desired number of partitions.
    - total_population: The total population across all nodes.
    
    Returns:
    - A combined metric score where lower is better, emphasizing population balance and compactness.
    """
    ideal_population = get_total_population(graph) / target_partitions
    actual_partitions = len(set(list(partition.values())))
    if actual_partitions != target_partitions:
        return ('Invalid partition')
    partition_pops = [0] * actual_partitions
    internal_distances = [0] * actual_partitions
    external_distances = [0] * actual_partitions
    internal_counts = [0] * actual_partitions
    external_counts = [0] * actual_partitions
    
    for node, district in partition.items():
        district = district - 1
        partition_pops[district] += graph.nodes[node]['population']
        for neighbor in graph.neighbors(node):
            if partition[node] == partition[neighbor]:
                internal_distances[district] += graph[node][neighbor]['weight']
                internal_counts[district] += 1
            else:
                external_distances[district] += graph[node][neighbor]['weight']
                external_counts[district] += 1
    
    pop_metric = [round(partition_pops[i]*100/ideal_population, 2) for i in range(actual_partitions)]
    
    # Distance metric calculation
    distance_metric = 0
    for i in range(actual_partitions):
        internal_mean = internal_distances[i] / internal_counts[i] if internal_counts[i] else 0
        external_mean = external_distances[i] / external_counts[i] if external_counts[i] else 1
        if internal_mean == 0:
            distance_metric += external_mean
        distance_metric += internal_mean / external_mean
    
    # Combine metrics (here simply summed, but you can adjust weighting)
    # combined_metric = pop_variance + distance_metric
    
    return (pop_metric, internal_mean)

def custom_metric(graph, partition, target_partitions):
    """
    Custom metric considering population distribution and internal/external mean distances.
    
    Parameters:
    - graph: The graph object.
    - partition: A dictionary mapping each node to its district.
    - target_partitions: The desired number of partitions.
    - total_population: The total population across all nodes.
    
    Returns:
    - A combined metric score where lower is better, emphasizing population balance and compactness.
    """
    ideal_population = get_total_population(graph) / target_partitions
    actual_partitions = len(set(list(partition.values())))
    partition_pops = [0] * actual_partitions
    internal_distances = [0] * actual_partitions
    external_distances = [0] * actual_partitions
    internal_counts = [0] * actual_partitions
    external_counts = [0] * actual_partitions
    
    for node, district in partition.items():
        district = district - 1
        partition_pops[district] += graph.nodes[node]['population']
        for neighbor in graph.neighbors(node):
            if partition[node] == partition[neighbor]:
                internal_distances[district] += graph[node][neighbor]['weight']
                internal_counts[district] += 1
            else:
                external_distances[district] += graph[node][neighbor]['weight']
                external_counts[district] += 1
    
    # Population MSE calculation
    pop_mse = np.mean([(pop - ideal_population) ** 2 for pop in partition_pops])
    # pop_variance = np.var([abs(pop - ideal_population) for pop in partition_pops])
    
    # Distance metric calculation
    distance_metric = 0
    for i in range(actual_partitions):
        internal_mean = internal_distances[i] / internal_counts[i] if internal_counts[i] else 0
        external_mean = external_distances[i] / external_counts[i] if external_counts[i] else 1
        if internal_mean == 0:
            distance_metric += external_mean
        distance_metric += internal_mean / external_mean
    
    # Combine metrics (here simply summed, but you can adjust weighting)
    # combined_metric = pop_variance + distance_metric
    
    return (pop_mse, distance_metric)

def compare_metrics(new, old):
    """
    Compare two metrics and return if new is better than old.
    """
    if new[0] <= old[0] and new[1] <= old[1]:
        return True
    pop_dif = new[0] - old[0]
    dist_dif = new[1] - old[1]
    pop_rel = pop_dif / old[0]
    dist_rel = dist_dif / old[1]
    if new[0] <= old[0] and dist_rel < 0.5:
        return True
    if new[1] <= old[1] and pop_rel < 0.5:
        return True
    return False

def aggregate_graph(graph, partition):
    """
    Aggregate the graph based on the current partition, where each community becomes a single node.
    
    Parameters:
    - graph: The original graph.
    - partition: A dictionary mapping each node to its community.
    
    Returns:
    - The aggregated graph where each node represents a community.
    """
    # Create a new graph where each node represents a community
    aggregated_graph = nx.Graph()
    
    # Map each node to its community and aggregate edges within communities
    community_map = {}
    for node, community in partition.items():
        if community not in community_map:
            community_map[community] = {
                'nodes': set(),
                'population': 0,
                'internal_edges': 0,
                'total_edges': 0
            }
        community_map[community]['nodes'].add(node)
        community_map[community]['population'] += graph.nodes[node]['population']
    
    # Add aggregated communities as nodes
    for community, info in community_map.items():
        aggregated_graph.add_node(community, population=info['population'])
    
    # Aggregate edges between communities
    for u, v, data in graph.edges(data=True):
        cu = partition[u]
        cv = partition[v]
        if cu == cv:
            # Internal edge, add to the internal edges count
            community_map[cu]['internal_edges'] += 1
        else:
            # Edge between communities, add or update edge in the aggregated graph
            if aggregated_graph.has_edge(cu, cv):
                aggregated_graph[cu][cv]['weight'] += data['weight']
            else:
                aggregated_graph.add_edge(cu, cv, weight=data['weight'])
                
    return aggregated_graph


def get_paths_between_communities(graph, partition):
    """
    Get list of all paths between communities in dictuionary with key as lenghts.
    """

    paths = {}
    for u, v, data in graph.edges(data=True):
        cu = partition[u]
        cv = partition[v]
        if cu != cv:
            length = data['weight']
            if length in paths:
                paths[length].append((cu, cv))
            else:
                paths[length] = [(cu, cv)]

    # Sort the dictionary by smallest index
    paths = dict(sorted(paths.items()))
    return paths

def update_partition_numbers(new_partition):
    # reset the partition numbers
    communities = list(set(new_partition.values()))
    for i, community in enumerate(communities):
        for node, com in new_partition.items():
            if community == com:
                new_partition[node] = i
    return new_partition

def aggregate_partitions(partition, paths):
    """
    Aggregate the partition based on the given list of paths (node to node).
    """
    new_partition = partition.copy()
    for u, v in paths:
        cu, cv = partition[u], partition[v]
        nodes_cu = [node for node, community in partition.items() if community == cu]
        for node in nodes_cu:
            new_partition[node] = cv

    return update_partition_numbers(new_partition)

def agrgeate_partitions_path(partition, path):
    """
    Aggregate the partition based on the given list of paths (node to node).
    """
    new_partition = partition.copy()
    u, v = path
    cu, cv = partition[u], partition[v]
    nodes_cu = [node for node, community in partition.items() if community == cu]
    for node in nodes_cu:
        new_partition[node] = cv

    return update_partition_numbers(new_partition)

def modified_louvain_algorithm2(G, target_partitions=40, max_iter=1000):
    
    # Initial partition with each node in its own community
    total_population = get_total_population(G)
    initial_partition = {node: node for node in G.nodes()}
    partition = initial_partition.copy()
    improvement = True
    iter_count = 0
    
    while (improvement or len(set(partition.values())) != target_partitions) and iter_count < max_iter:
        iter_count += 1
        improvement = True
        
        # Evaluate current partition
        current_metric = custom_metric(G, partition, target_partitions)
        
        # Attempt to optimize partition
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                # Tentatively move node to neighbor's partition for evaluation
                original_partition = partition[node]
                partition[node] = partition[neighbor]
                
                new_metric = custom_metric(G, update_partition_numbers(partition), target_partitions)
                
                # Revert if no improvement
                if not compare_metrics(new_metric, current_metric):
                    partition[node] = original_partition
                else:
                    current_metric = new_metric
                    partition = update_partition_numbers(partition)
                    improvement = False
        
        # Aggregate graph based on current partition if there was an improvement
        if improvement:
            inter_paths = get_paths_between_communities(G, partition)
            for length, paths in inter_paths.items():
                new_partition = aggregate_partitions(partition, paths)
                new_metric = custom_metric(G, new_partition, target_partitions)
                if compare_metrics(new_metric, current_metric):
                    partition = new_partition
                    improvement = False
                    break
            # This step would involve your logic for path selection and making internal paths based on shortest distances
            
            # Further optimization on the aggregated graph could go here
            # Note: For actual pathfinding and optimization, additional logic is needed
            
    return partition

def reache_target(partition, target):
    return len(set(partition.values())) == target

def compare_list_metrics(old, new_list):
    good_metrics = []
    for new in new_list:
        if compare_metrics(new, old):
            good_metrics.append(new)
    if not good_metrics:
        return old
    relative_metrics = [((old[0] - new[0]) / old[0], (old[1] - new[1]) / old[1]) for new in good_metrics]
    best_metric = good_metrics[np.argmax([sum(rel) for rel in relative_metrics])]
    best_metric_index = new_list.index(best_metric)
    return best_metric_index

def compare_list_metrics_all_good(old, new_list):
    good_metrics = []
    for new in new_list:
        if compare_metrics(new, old):
            good_metrics.append(new)
    if not good_metrics:
        return None
    indices_good_metrics = [new_list.index(good) for good in good_metrics]
    return indices_good_metrics


def modified_louvain_algorithm3(G, target_partitions=40, max_iter=1000):
    
    # Initial partition with each node in its own community
    initial_partition = {node: node for node in G.nodes()}
    partition = initial_partition.copy()
    improvement = True
    iter_count = 0
    
    while not reache_target(partition, target_partitions) and iter_count < max_iter:
        print(len(set(partition.values())))
        iter_count += 1        
        # Evaluate current partition
        current_metric = custom_metric(G, partition, target_partitions)

        inter_paths = get_paths_between_communities(G, partition)
        first_lengths = list(inter_paths.keys())[:3]
        first_paths = [inter_paths[length] for length in first_lengths][0]
        new_partitions = [agrgeate_partitions_path(partition, path) for path in first_paths]
        new_metrics = [custom_metric(G, new_partition, target_partitions) for new_partition in new_partitions]
        new_partition = new_partitions[compare_list_metrics(current_metric, new_metrics)]
        if new_partition == partition:
            improvement = False
        else:
            partition = new_partition
            current_metric = new_metrics[compare_list_metrics(current_metric, new_metrics)]
            improvement = True

        if not improvement:
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    # Tentatively move node to neighbor's partition for evaluation
                    original_partition = partition[node]
                    partition[node] = partition[neighbor]
                    
                    new_metric = custom_metric(G, update_partition_numbers(partition), target_partitions)
                    
                    # Revert if no improvement
                    if not compare_metrics(new_metric, current_metric):
                        partition[node] = original_partition
                    else:
                        current_metric = new_metric
                        partition = update_partition_numbers(partition)
                        improvement = True
        
        if not improvement:
            partition = new_partitions[0]
            current_metric = new_metrics[0]
            
    return partition

def change_partitions(partition, node, neighbour):
    cn = partition[node]
    cv = partition[neighbour]
    new_partition = partition.copy()
    for node, com in new_partition.items():
        if com == cn:
            new_partition[node] = cv
    return new_partition


def modified_louvain_algorithm4(G, target_partitions=40, max_iter=1000):
    
    # Initial partition with each node in its own community
    initial_partition = {node: node for node in G.nodes()}
    partition = initial_partition.copy()
    improvement = True
    iter_count = 0
    
    while not reache_target(partition, target_partitions) and iter_count < max_iter:
        print(len(set(partition.values())))
        iter_count += 1        
        # Evaluate current partition
        current_metric = custom_metric(G, partition, target_partitions)

        inter_paths = get_paths_between_communities(G, partition)
        first_lengths = list(inter_paths.keys())[:2]
        first_paths = [inter_paths[length] for length in first_lengths][0]
        new_partitions = [agrgeate_partitions_path(partition, path) for path in first_paths]
        new_metrics = [custom_metric(G, new_partition, target_partitions) for new_partition in new_partitions]
        indicies_good_partitions = compare_list_metrics_all_good(current_metric, new_metrics)
        if indicies_good_partitions:
            good_paths = [first_paths[i] for i in indicies_good_partitions]
            for path in good_paths:
                new_partition = agrgeate_partitions_path(partition, path)
                new_metric = custom_metric(G, new_partition, target_partitions)
                partition = new_partition
                current_metric = new_metric
                improvement = True
        else:
            improvement = False

        if not improvement:
            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    # Tentatively move node to neighbor's partition for evaluation
                    new_partition = change_partitions(partition, node, neighbor)
                    # original_partition = partition[node]
                    # partition[node] = partition[neighbor]
                    
                    new_metric = custom_metric(G, update_partition_numbers(new_partition), target_partitions)
                    
                    # Revert if no improvement
                    print('here')
                    if compare_metrics(new_metric, current_metric):
                        if partition[node] != partition[neighbor]:
                            current_metric = new_metric
                            partition = update_partition_numbers(new_partition)
                            improvement = True
        
        if not improvement:
            partition = new_partitions[0]
            current_metric = new_metrics[0]
            
    return partition


partition = modified_louvain_algorithm4(G, 10)

# pop_variance, distric_dist, district_pops = evaluate_districts(G, partition)

# print(pop_variance, distric_dist, district_pops)

print(evaluation(G, partition, 10))

