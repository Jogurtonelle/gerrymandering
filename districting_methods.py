from copy import deepcopy
from math import exp, sqrt
import networkx as nx
import numpy as np
import random
import time
import threading
import pickle as pkl
import geopandas as gpd
import concurrent.futures

def seeding_algorithm(graph: nx.Graph,
                      number_of_districts: int,
                      deviation: float = 0.01,
                      retries: int = 100,
                      seed: int = int(time.time()),
                      ) -> list:
    """
    Seeding algorithm for districting.

    Parameters
    ----------
    graph : networkx.Graph
        A graph representing the counties and their neighbors.
        Each node in the graph needs to have a 'voters' attribute representing the number of voters in the county.

    number_of_districts : int
        The number of districts to create.

    deviation : float, optional
        The deviation from the average number of voters in a district. 
        For 5% deviation, use 0.05. Default is 0.01 (1%).
        Derivation has to be between 0 and 1 (0% and 100%).

    retries : int, optional
        The number of retries if the algorithm fails to create districts. Default is 100.

    seed : int, optional
        The seed for the random number generator. Default is the current time.

    Returns
    -------
    list
        Array of districts. Each district is a list of counties' IDs. 
        For a district with one county, the list will contain one element.

    References
    ----------
    [1] https://doi.org/10.1016/S0962-6298(99)00047-5
    """

    if deviation > 1 or deviation < 0:
            raise ValueError("Derivation has to be between 0 and 1")
    
    if number_of_districts < 1 or number_of_districts > len(graph.nodes):
        raise ValueError("Number of districts has to be greater than 0 and less than the number of nodes in the graph")
    
    if not all('voters' in graph.nodes[node] for node in graph.nodes):
        raise ValueError("Each node in the graph has to have a 'voters' attribute")
    
    if retries < 1:
        raise ValueError("Number of retries has to be greater than 0")

    def seeding_algorithm_inner(graph: nx.Graph,
                                number_of_districts: int,
                                deviation: float = 0.01,
                                seed: int = int(time.time()),
                                ) -> list:
        """
        Helper function for seeding_algorithm (to prevent isolation of nodes - if occurs, then rerun the seeding algorithm)
        """
        graph_copy = graph.copy()

        #Calculate the average number of voters in a district
        total_voters = sum(graph_copy.nodes[node]['voters'] for node in graph_copy.nodes)
        
        try:
            average_voters = int(total_voters / number_of_districts)
        except ZeroDivisionError:
            raise ValueError("Number of districts has to be greater than 0")

        #Every county with more than the average number of voters (+/- deviation) is considered a district and is removed from the graph
        districts = []
        nodes_to_remove = [
            node for node in graph_copy.nodes 
            if graph_copy.nodes[node]['voters'] > average_voters * (1 + abs(deviation))
        ]

        for node in nodes_to_remove:
            districts.append([node])
        graph_copy.remove_nodes_from(nodes_to_remove)

        number_of_districts -= len(districts)
        random.seed(seed)
        #Creating the remaining districts as long as there are nodes in the graph
        while len(graph_copy.nodes) > number_of_districts and len(graph_copy.nodes) > 0 and number_of_districts > 0:
            district = []
            #Start district from a random county
            district.append(random.choice(list(graph_copy.nodes)))
            current_voters = graph_copy.nodes[district[0]]['voters']
            neighbors = set(graph_copy.neighbors(district[0]))

            #Add neighbors to the district until the number of voters is close to the average
            while current_voters < average_voters * (1 - deviation) and len(graph_copy.nodes) > number_of_districts:
                neighbors.difference_update(district)  # Remove nodes that are already in the district
                
                if not neighbors:
                    break # No neighbors left to add

                #Remove nodes that have too many voters
                neighbors_copy = neighbors.copy()
                for neighbor in neighbors_copy:
                    if graph_copy.nodes[neighbor]['voters'] + current_voters > average_voters * (1 + deviation):
                        neighbors.remove(neighbor)

                #If there are no neighbors left, add the one that is the closest to the average number of voters
                #if the average number of voters is closer to the current number of voters with the best neighbor
                if not neighbors:
                    best_neighbor = None
                    best_diff = abs(average_voters - current_voters)
                    for neighbor in neighbors_copy:
                        diff = abs(graph_copy.nodes[neighbor]['voters'] + current_voters - average_voters)
                        if diff < best_diff:
                            best_diff = diff
                            best_neighbor = neighbor
                    
                    if best_neighbor is not None:
                        district.append(best_neighbor)
                        current_voters += graph_copy.nodes[best_neighbor]['voters']

                    break
                
                #Try to find a neighbor most compact to the rest of the district
                best_neighbor = None
                best_compactness = 0
                for neighbor in neighbors:
                    compactness = 0
                    for node in district:
                        compactness += 1 if neighbor in graph_copy.neighbors(node) else 0
                    if compactness > best_compactness:
                        best_compactness = compactness
                        best_neighbor = neighbor
                
                #Add the best neighbor to the district
                if best_neighbor is None:
                    break

                district.append(best_neighbor)
                current_voters += graph_copy.nodes[best_neighbor]['voters']
                neighbors.remove(best_neighbor)
                neighbors.update(graph_copy.neighbors(best_neighbor))
            
            #Remove the district from the graph
            for node in district:
                graph_copy.remove_node(node)
            
            #Add the district to the list of districts
            districts.append(district)
            number_of_districts -= 1
            

        #Add the remaining nodes to the districts 
        if len(graph_copy.nodes) > 0:
            for node in graph_copy.nodes:
                #select a random neighbour of node
                neighbor = random.choice(list(graph.neighbors(node)))

                #add node to the district of the neighbor
                for district in districts:
                    if neighbor in district:
                        district.append(node)
                        break
        
        #Add nodes that during the process were made isolated
        covered_nodes = {node for district in districts for node in district}
        missing_nodes = set(graph.nodes) - covered_nodes
        prev_len_missing_nodes = len(missing_nodes)

        # if len(missing_nodes) > 0.1 * len(graph.nodes):
        #     print (len(missing_nodes)/len(graph.nodes))
        #     return [], True
        
        while missing_nodes:
            for node in missing_nodes:
                #if the node is not connected to any of the covered nodes, skip it for now (it will be added later)
                if not any(neighbor in covered_nodes for neighbor in graph.neighbors(node)):
                    continue

                #select a random neighbour of node
                finish = False
                i = 0
                while not finish:
                    neighbor = random.choice(list(graph.neighbors(node)))
                    if neighbor in covered_nodes:
                        #add node to the district of the neighbor
                        for district in districts:
                            if neighbor in district:
                                district.append(node)
                                covered_nodes.add(node)
                                finish = True
                                break
                    i += 1
                    if i > 50:
                        #Probability that this happens is too low to be considered, so it is an error
                        raise ValueError("Unexpected error in seeding algorithm - node is not connected to any of the covered nodes")

            missing_nodes = set(graph.nodes) - covered_nodes
            if prev_len_missing_nodes == len(missing_nodes):
                raise ValueError("Unexpected error in seeding algorithm - missing nodes are not changing")

        #If there are districts with only one node, add them to the list of districts
        if number_of_districts > 0:
            for node in graph_copy.nodes:
                districts.append([node])

        return districts, False
 
    for i in range(retries):
        try:
            result, try_again = seeding_algorithm_inner(graph, number_of_districts, deviation, seed)
            if try_again:
                deviation *= 1.5
            else:
                return result
        except ValueError as e:
            print(f"Error in seeding algorithm: {e}. Retrying...")
            seed += 1

    raise ValueError("Unexpected error in seeding algorithm - try again with different parameters")

def graph_cut_algorithm(graph: nx.Graph,
                        number_of_districts: int,
                        deviation: float = 0.01,
                        lambda_val: int = 10,
                        beta_start: int = 1,
                        beta_end: int = 10,
                        q: float = 0.05,
                        num_of_chains: int = 5,
                        max_iterations: int = 200,
                        T: int = 100,
                        seed: int = int(time.time()),
                        ) -> list:
    """
    Sampling contiguous redistricting plans using Markov Chain Monte Carlo.

    Parameters
    ----------
    graph : networkx.Graph
        A graph representing the geographical units.
        Each node should have a 'voters' attribute.

    number_of_districts : int
        Number of districts to form.

    deviation : float, optional
        Acceptable deviation from equal population. Default is 0.01 (1%).

    q : float, optional
        Probability to "turn on" edges. Default is 0.05.

    max_iterations : int, optional
        Maximum number of MCMC iterations. Default is 1000.

    seed : int, optional
        Random seed. Default is the current time.

    Returns
    -------
    list
        List of districts, where each district is a list of node IDs.
    """

    #Helper functions to run the algorithm in a separate thread
    def turn_on_edges(edge_set, current_districts, q):
            for subgraph in current_districts:
                for node in subgraph.nodes:
                    for neighbor in graph.neighbors(node):
                        if neighbor in subgraph.nodes and random.random() < q and (node, neighbor) not in edge_set and (neighbor, node) not in edge_set:
                            edge_set.add((node, neighbor))

    def identify_boundary_nodes(current_districts, boundary_nodes):
            for subgraph in current_districts:
                for node in subgraph.nodes:
                    if any(neighbor not in subgraph.nodes for neighbor in graph.neighbors(node)):
                        boundary_nodes.add(node)

    def g_beta(districts, beta):
        result = 0
        for district in districts:
            result += (sum(graph.nodes[node]['voters'] for node in district)/average_voters) - 1
        return np.exp(-beta * result)

    random.seed(seed)

    cumulative_distributions = [0] * (len(graph.nodes) + 1)
    result = lambda_val
    temp = lambda_val
    cumulative_distributions[0] = np.exp(-lambda_val) * result
    for i in range(1, len(graph.nodes) + 1):
        temp *= (lambda_val / i)
        result += temp
        cumulative_distributions[i] = np.exp(-lambda_val) * result

    # Initial districts from the seeding algorithm (pi_0)
    temp = True
    districts = []
    while temp:
        try:
            districts = seeding_algorithm(graph, number_of_districts, deviation, seed)
            temp = False
        except ValueError as e:
            seed += 1

    average_voters = sum(graph.nodes[node]['voters'] for node in graph.nodes) / number_of_districts

    # Convert initial districts into subgraph objects
    current_districts = []
    for district in districts:
        subgraph = nx.Graph()
        for node in district:
            subgraph.add_node(node, voters=graph.nodes[node]['voters'])
            for neighbor in graph.neighbors(node):
                if neighbor in district:
                    subgraph.add_edge(node, neighbor)
        current_districts.append(subgraph)

    #make dictonary there the key is the node and the value is the subdistrict id in subgraphs list
    node_to_subgraph = {node: i for i, subgraph in enumerate(current_districts) for node in subgraph.nodes}
    
    def main_algorithm (beta : int, 
                        iterations : int,
                        _current_districts : list,
                        _node_to_subgraph : dict,
                        ):
        for iteration in range(iterations):
            # Step 1: "Turn on" edges
            edge_set = set()
            turn_on_edges(edge_set, _current_districts, q)
                
            # Step 2: Identify boundary connected components
            boundary_nodes = set()

            identify_boundary_nodes(_current_districts, boundary_nodes)

            boundary_components = [] #B(CP, pi_t) in the paper
            visited = set()
            for node in boundary_nodes:
                if node not in visited:
                    component = set()
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            component.add(current)
                            stack.extend(
                                neighbor for neighbor in graph.neighbors(current)
                                if neighbor not in visited and ((current, neighbor) in edge_set or (neighbor, current) in edge_set)
                            )
                    boundary_components.append(component)

            # Step 3: Select nonadjacent connected components
            B_CP_pi = boundary_components.copy()
            R = -1 
            while R < 0 or R >= len(boundary_components):
                R = int(np.random.poisson(lam=lambda_val))

            print(f"R: {R}", "Boundary components:", len(boundary_components))

            selected_components = [] #V_cp in the paper
            while len(selected_components) < R:
                component = random.choice(boundary_components)

                if component in selected_components:
                    continue
                
                #check if component is adjacent to another element of V_cp by checking if there is an edge between the two components in the graph
                adjacent = False
                for selected_component in selected_components:
                    if any(graph.has_edge(node1, node2) for node1 in component for node2 in selected_component):
                        adjacent = True
                        break

                #check if the removal of V_cp results in a noncontiguous district
                if not adjacent:
                    #find from which district the component is
                    component_district = _node_to_subgraph.get(next(iter(component)), None)
                    if component_district is not None:
                        #remove the component from the district
                        district = _current_districts[component_district].copy()
                        for node in component:
                            district.remove_node(node)
                        #check if the district is still contiguous
                        if nx.number_connected_components(district) == 1:
                            selected_components.append(component)
                        else:
                            boundary_components.remove(component)
                            if len(boundary_components) <= R:
                                R = len(boundary_components) - 1
            V_cp = selected_components.copy()

            # Step 4: Propose district swaps
            new_districts = deepcopy(_current_districts)
            proposed_node_to_subgraph = _node_to_subgraph.copy()

            for component in selected_components:
                #get a random district to move the component to, which is connected to the component
                connected_districts = {_node_to_subgraph[neighbor] for node in component for neighbor in graph.neighbors(node) if neighbor in _node_to_subgraph}

                #remove the district from the connected_districts set
                current_district = _node_to_subgraph.get(next(iter(component)))
                connected_districts.discard(current_district)

                #if there are no connected districts, skip the component (in typical case, this should not happen)
                if len(connected_districts) == 0:
                    continue

                #get the district to move the component to
                new_district = random.choice(list(connected_districts))

                #move the component to the proposed district
                for node in component:
                    new_districts[current_district].remove_node(node)
                    new_districts[new_district].add_node(node, voters=graph.nodes[node]['voters'])
                    proposed_node_to_subgraph.update({node: new_district})

                #update connections in the proposed district according to the graph
                for node in component:
                    for neighbor in graph.neighbors(node):
                        if neighbor in new_districts[new_district]:
                            new_districts[new_district].add_edge(node, neighbor)

            # Step 5: Accept or reject the proposal

            #get B_CP_pi_prime
            boundary_nodes_prime = set()

            identify_boundary_nodes(new_districts, boundary_nodes_prime)
            
            B_CP_pi_prime = []

            visited = set()
            for node in boundary_nodes_prime:
                if node not in visited:
                    component = set()
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            component.add(current)
                            stack.extend(
                                neighbor for neighbor in graph.neighbors(current)
                                if neighbor not in visited and ((current, neighbor) in edge_set or (neighbor, current) in edge_set)
                            )
                    B_CP_pi_prime.append(component)

            F_B_CP_pi = cumulative_distributions[len(B_CP_pi)]
            F_B_CP_pi_prime = cumulative_distributions[len(B_CP_pi_prime)]

            #Get |C(π, V_CP)| and |C(π', V_CP)|
            C_pi_V_cp = set()
            C_pi_prime_V_cp = set()

            #unpack V_cp to get all nodes in the components
            nodes_vcp = set()
            for component in V_cp:
                nodes_vcp.update(component)

            for node in nodes_vcp:
                #C_pi_V_cp
                district = _node_to_subgraph.get(node, None)
                if district is None:
                    raise ValueError("Node is not in any district")
                for neighbor in _current_districts[district].neighbors(node):
                    if neighbor not in nodes_vcp:
                        C_pi_V_cp.add((node, neighbor))

                #C_pi_prime_V_cp
                district = proposed_node_to_subgraph.get(node, None)
                if district is None:
                    raise ValueError("Node is not in any district")
                for neighbor in new_districts[district].neighbors(node):
                    if neighbor not in nodes_vcp:
                        C_pi_prime_V_cp.add((node, neighbor))

            #get the acceptance probability
            acceptance_probability = ((len(B_CP_pi)/len(B_CP_pi_prime))**R) * (F_B_CP_pi/F_B_CP_pi_prime) * ((1-q)**(len(C_pi_prime_V_cp) - len(C_pi_V_cp))) * (g_beta(new_districts, beta) / g_beta(_current_districts, beta))
            print(f"Acceptance probability: {acceptance_probability}")
            acceptance_probability = min(1, acceptance_probability)

            if random.random() <= acceptance_probability:
                _current_districts.clear()
                _current_districts.extend(new_districts)

                _node_to_subgraph.clear()
                _node_to_subgraph.update(proposed_node_to_subgraph)

    chain_threads = []
    chain_current_districts = [deepcopy(current_districts) for i in range(num_of_chains * (beta_end - beta_start + 1))]
    chain_node_to_subgraph = [node_to_subgraph.copy() for i in range(num_of_chains * (beta_end - beta_start + 1))]
    chain_data = []

    index = 0
    for i in range(beta_start, beta_end + 1):
        for j in range(num_of_chains):
            chain_data.append(i)
            chain_threads.append(threading.Thread(target=main_algorithm, args=(i, min(T, max_iterations), chain_current_districts[index], chain_node_to_subgraph[index])))
            index += 1

    for chain in chain_threads:
        chain.start()

    for chain in chain_threads:
        chain.join()

    iterations = T
    while iterations < max_iterations:
        j = random.randint(0, num_of_chains * (beta_end - beta_start + 1) - 1)
        k = random.randint(0, num_of_chains * (beta_end - beta_start + 1) - 1)

        if j == k:
            k = (k + 1) % (num_of_chains * (beta_end - beta_start + 1))

        gamma = min(1, (g_beta(chain_current_districts[j], chain_data[k]) * g_beta(chain_current_districts[k], chain_data[j])) / (g_beta(chain_current_districts[j], chain_data[j]) * g_beta(chain_current_districts[k], chain_data[k])))
        print(f"Gamma: {gamma}")
        if random.random() < gamma:
            chain_data[j], chain_data[k] = chain_data[k], chain_data[j]
        
        for i in range(num_of_chains * (beta_end - beta_start + 1)):
            chain_threads[i] = threading.Thread(target=main_algorithm, args=(chain_data[i], min(T, max_iterations - iterations), chain_current_districts[i], chain_node_to_subgraph[i]))
        
        for chain in chain_threads:
            chain.start()

        for chain in chain_threads:
            chain.join()

        iterations += T


    #get the final districts with the smallest variance of voters
    best_proposal_index = -1
    worst_proposal_index = -1
    best_proposal_score = np.var([sum(graph.nodes[node]['voters'] for node in district.nodes) for district in current_districts])
    worst_proposal_score = np.inf
    print(f"Starting score: {best_proposal_score}")
    for i, proposal in enumerate(chain_current_districts):
        score = np.var([sum(graph.nodes[node]['voters'] for node in district.nodes) for district in proposal])
        print(f"Chain {i} score: {score}")
        if score < best_proposal_score:
            best_proposal_score = score
            best_proposal_index = i
        if score > worst_proposal_score:
            worst_proposal_index = i
            worst_proposal_score = score

    districts = []

    if best_proposal_index == -1:
        for district in current_districts:
            districts.append(list(district.nodes))
    else:
        for district in chain_current_districts[best_proposal_index]:
            districts.append(list(district.nodes))

    districts = []
    for district in chain_current_districts[worst_proposal_index]:
        districts.append(list(district.nodes))
    
    return districts, districts

def merge_min_cut_max(graph: nx.Graph,
                      number_of_districts: int,
                      dist_dict: dict,
                      dist_population_dict: dict):

    avg_voters = sum(graph.nodes[node]['voters'] for node in graph.nodes) / number_of_districts

    while any(pop < avg_voters * 0.3 for pop in dist_population_dict.values()): #and any(pop > avg_voters * 1.7 for pop in dist_population_dict.values()):
        smallest_district_id = min(dist_population_dict, key=dist_population_dict.get)
        largest_district_id = max(dist_population_dict, key=dist_population_dict.get)
        
        #get neighbors of the smallest district
        smallest_district_nodes = {node for node in dist_dict if dist_dict[node] == smallest_district_id}
        smallest_district_neighbors = set()
        for node in smallest_district_nodes:
            smallest_district_neighbors.update(graph.neighbors(node))

        smallest_district_neighbors.difference_update(smallest_district_nodes)  # Remove nodes that are already in the district

        if not smallest_district_neighbors:
            break

        # Find the neighbor with the smallest population
        smallest_neighbor = min(smallest_district_neighbors, key=lambda x: graph.nodes[x]['voters'])


        # divide the largest district into two districts
        largest_district_nodes = {node for node in dist_dict if dist_dict[node] == largest_district_id}
        if len(largest_district_nodes) <= 1:
            break

        largest_district_subgraph = nx.Graph()
        largest_district_subgraph.add_nodes_from(largest_district_nodes)
        for node in largest_district_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor in largest_district_nodes:
                    largest_district_subgraph.add_edge(node, neighbor)

        largest_district_subgraph_population = sum(graph.nodes[node]['voters'] for node in largest_district_nodes)

        # Create a subgraph with randomly selected nodes from the largest district
        for i in range (20):  # Retry up to 20 times to find a connected subgraph
            # Randomly select a boundary node from the largest district
            boundary_nodes = [node for node in largest_district_nodes if any(neighbor not in largest_district_nodes for neighbor in graph.neighbors(node))]
            random_node = random.choice(boundary_nodes)

            # Create a subgraph  from the largest district nodes
            largest_district_subgraph_1 = largest_district_subgraph.copy()
            largest_district_subgraph_1.remove_node(random_node)

            largest_district_subgraph_2 = nx.Graph()
            largest_district_subgraph_2.add_node(random_node)

            sub_2_neighbors = [neighbor for neighbor in graph.neighbors(random_node) if neighbor in largest_district_nodes]
            sub_2_population = graph.nodes[random_node]['voters']

            while nx.is_connected(largest_district_subgraph_1) and sub_2_neighbors and sub_2_population < 0.4 * largest_district_subgraph_population:
                # Move a neighbor from sub_1 to sub_2
                neighbor = sub_2_neighbors.pop()
                largest_district_subgraph_1.remove_node(neighbor)
                largest_district_subgraph_2.add_node(neighbor)
                largest_district_subgraph_2.add_edge(random_node, neighbor)
                sub_2_population += graph.nodes[neighbor]['voters']
                sub_2_neighbors.extend(
                    n for n in graph.neighbors(neighbor) if n in largest_district_nodes and n not in largest_district_subgraph_1.nodes and n not in largest_district_subgraph_2.nodes
                )

            if nx.is_connected(largest_district_subgraph_1) and nx.is_connected(largest_district_subgraph_2):
                break

        else:
            print("Failed to divide the largest district into two connected subgraphs after 20 attempts.")
            break

        # Merge the smallest district into the smallest neighbor
        dist_dict.update({node: dist_dict[smallest_neighbor] for node in smallest_district_nodes})
        dist_population_dict[dist_dict[smallest_neighbor]] += dist_population_dict[smallest_district_id]
        del dist_population_dict[smallest_district_id]

        # Assign smallest_district_id to the first subgraph and largest_district_id to the second subgraph
        for node in largest_district_subgraph_1.nodes:
            dist_dict[node] = smallest_district_id
        for node in largest_district_subgraph_2.nodes:
            dist_dict[node] = largest_district_id

        # Update the population dictionary
        dist_population_dict[smallest_district_id] = sum(graph.nodes[node]['voters'] for node in largest_district_subgraph_1.nodes)
        dist_population_dict[largest_district_id] = sum(graph.nodes[node]['voters'] for node in largest_district_subgraph_2.nodes)

    return dist_dict, dist_population_dict

def redist_flip_alg(graph: nx.Graph,
                    polygon_gdf: gpd.GeoDataFrame,
                    border_geometry: gpd.GeoDataFrame,
                    number_of_districts: int,
                    hot_steps: int = 200,
                    annealing_steps: int = 100,
                    cold_steps: int = 400,
                    lambda_prob: float = 0.01,
                    beta_eq_pop_target: float = 0.5,
                    beta_compactness_target: float = 0.5,
                    initial_seeding_attempts: int = 20,
                    initial_districts: list = None,
                    seed: int = int(time.time()),
                    ) -> list:
    """
    Redistricting algorithm by flipping edges, using simulated annealing.
    
    Parameters
    ----------
    graph : networkx.Graph
        A graph representing the counties and their neighbors.
        Each node in the graph needs to have a 'voters' attribute representing the number of voters in the county.

    polygon_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing the geometries of the counties in format:
        {'id': int/string, 'geometry': Polygon}. IDs should match the node IDs in the graph.

    border_geometry : geopandas.GeoDataFrame
        A GeoDataFrame containing the geometry of the country's border.
        This is used in calculating compactness of the districts - if district is by the border, its compactness can be lowered by the border shape.

    number_of_districts : int
        The number of districts to create.

    hot_steps : int, optional
        The number of steps at the beginning of the algorithm where the acceptance rate is 100%. Default is 200 steps.

    annealing_steps : int, optional
        The number of steps where the temperature is decreasing. Default is 100 steps.

    cold_steps : int, optional
        The number of steps at the end of the algorithm where the temperature is 0. Default is 400 steps.

    lambda_prob : float, optional
        The probability of flipping an edge. Default is 1%.
        Value has to be between 0 and 1.

    beta_eq_pop_target : float, optional
        Parameter for the equal population term in the objective function. Default is 0.5.

    beta_compactness_target : float, optional
        Parameter for the compactness term in the objective function. Default is 0.5.

    seed : int, optional
        The seed for the random number generator. Default is the current time.

    Returns
    -------
    list
        Array of districts. Each district is a list of counties' IDs (acoording to the graph).
        For a district with one county, the list will contain one element.

    References
    ----------
    [1] https://doi.org/10.1016/S0962-6298(99)00047-5 #TODO - to nie jest poprawny link
    """

    # Check input parameters
    if not isinstance(graph, nx.Graph):
        raise TypeError("Graph must be a networkx.Graph object")
    if not isinstance(polygon_gdf, gpd.GeoDataFrame):
        raise TypeError("polygon_gdf must be a geopandas.GeoDataFrame object")
    if not isinstance(border_geometry, gpd.GeoDataFrame):
        raise TypeError("border_geometry must be a geopandas.GeoDataFrame object")
    else:
        border_geometry = border_geometry.union_all()  # Ensure border geometry is a single polygon
    if not all(node in graph.nodes for node in polygon_gdf['graph_node_id']):
        raise ValueError("All IDs in polygon_gdf must match the node IDs in the graph")
    if lambda_prob < 0 or lambda_prob > 1:
        raise ValueError("Lambda probability has to be between 0 and 1")
    if number_of_districts < 1:
        raise ValueError("Number of districts has to be greater than 0")
    if hot_steps < 1:
        raise Warning("Hot steps should be greater than 0, but if you want to skip the hot steps, you're free to do so")
        hot_steps = 0
    if cold_steps < 1:
        raise Warning("Cold steps should be greater than 0, but if you want to skip the hot steps, you're free to do so")
        cold_steps = 0
    if annealing_steps < 1:
        raise ValueError("Annealing steps should be greater than 0")
        annealing_steps = 0

    avg_voters = sum(graph.nodes[node]['voters'] for node in graph.nodes) / number_of_districts

    # Seeding and Initialization - Find the best initial seeding guess
    best_score = np.inf
    if initial_districts is not None:
        initial_seeding_attempts = 0
    else:
        print(f"Finding the best initial seeding out of {initial_seeding_attempts} attempts...")
    for i in range(initial_seeding_attempts):
        temp = seeding_algorithm(graph, number_of_districts, seed = seed + i)

        # Calculate the score for the current seeding
        dist_dict = {node: i for i, dist in enumerate(temp) for node in dist}  # node to district mapping
        dist_population_dict = {i: sum(graph.nodes[node]['voters'] for node in dist) for i, dist in enumerate(temp)}  # district to population mapping
        pop_max_deriv = max(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1))
        pop_avg_deriv = np.mean(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1) ** 2)

        districts = [[node for node in dist_dict if dist_dict.get(node) == i] for i in range(number_of_districts)]
        dist_gdf = gpd.GeoDataFrame({
            'id': range(number_of_districts),
            'geometry': [polygon_gdf[polygon_gdf['graph_node_id'].isin(dist)].union_all() for dist in districts],
            'compactness': [0] * number_of_districts
        })

        # Calculate compactness for each district - area / area of (the convex hull intersected with the border)
        dist_gdf['compactness'] = dist_gdf.apply(
            lambda row: row['geometry'].area / (row['geometry'].convex_hull.intersection(border_geometry).area),
            axis=1
        )

        compactness = dist_gdf['compactness'].mean()  # Average compactness across all districts

        score = beta_eq_pop_target * pop_avg_deriv + beta_compactness_target * (1 - compactness)
        #print(f"Seeding attempt {i+1}/{initial_seeding_attempts} - score: {score}, pop_eq: {pop_avg_deriv}, compactness: {compactness}")
        if score < best_score:
            best_score = score
            initial_districts = temp

    # Initial districts - convert to a dictionary mapping nodes to districts
    dist_dict = {node: i for i, dist in enumerate(initial_districts) for node in dist} # node to district mapping
    dist_population_dict = {i: sum(graph.nodes[node]['voters'] for node in dist) for i, dist in enumerate(initial_districts)} # district to population mapping
    
    pop_max_deriv = max(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1))
    pop_avg_deriv = np.mean(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1))

    if random.random() < 0.5:
        dist_dict, dist_population_dict = merge_min_cut_max(graph, number_of_districts, dist_dict, dist_population_dict)

    # print("Old Population Max Derivative:", pop_max_deriv)
    # print("Old Population Average Derivative:", pop_avg_deriv)

    # Recalculate population derivatives
    pop_max_deriv = max(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1))
    pop_avg_deriv = sum(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1) ** 2 )

    # print("New Population Max Derivative:", pop_max_deriv)
    # print("New Population Average Derivative:", pop_avg_deriv)

    # Create a GeoDataFrame for districts - storing the geometries of the districts
    dist_gdf = gpd.GeoDataFrame()

    #Getting boundary nodes - v such that N(v) & S_δ(S_delta) is not empty
    not_active_edges = {(u, v) for u, v in graph.edges() if dist_dict.get(u) != dist_dict.get(v)}
    active_edges = set(graph.edges) - not_active_edges
    boundary_nodes = {node for edge in not_active_edges for node in edge} # Nodes that are on the boundary of districts (connected to nodes in different districts)
    
    pop_max_deriv_proposal = 0
    pop_avg_deriv_proposal = 0
    compactness_proposal = 0
    pop_eq_beta, comp_beta = 0, 0
    compactness = 0
    random.seed(seed)

    # Information for the user
    rejected_count = 0
    accepted_count = 0
    progress_data = np.zeros((int((annealing_steps + cold_steps + hot_steps)/100), 5))
    print(f"Starting redistricting algorithm with {number_of_districts} districts, hot steps: {hot_steps}, annealing steps: {annealing_steps}, cold steps: {cold_steps}, lambda probability: {lambda_prob}, beta_eq_pop_target: {beta_eq_pop_target}, beta_compactness_target: {beta_compactness_target}")
    
    step = 1
    while step < hot_steps + annealing_steps + cold_steps:

        # Create a copy of the current district graph
        dist_dict_proposal = dist_dict.copy()
        dist_population_dict_proposal = dist_population_dict.copy()
        
        if step > hot_steps:
            dist_gdf_proposal = dist_gdf.copy()
            
        #set flipped edges
        flipped_edges = {edge for edge in active_edges if random.random() <= lambda_prob}
        if not flipped_edges:
            continue

        # Identify boundary-connected components
        temp_graph = nx.Graph(flipped_edges)
        boundary_components = [comp for comp in nx.connected_components(temp_graph) if any(node in boundary_nodes for node in comp)]
        if not boundary_components:
            continue
        
       #select a subset of boundary components not connecting to each other
        R = max(min(len(boundary_components) - 1, np.random.normal(0.001*len(boundary_components), 0.001*len(boundary_components))), 1)
        selected_components, selected_nodes = [], set()
        while len(selected_components) < R and boundary_components:
            component = random.choice(list(boundary_components))
            if component not in selected_components and not any(graph.has_edge(node1, node2) for node1 in component for node2 in selected_nodes):
                selected_components.append(component)
                selected_nodes.update(component)
                boundary_components.remove(component)

        #move the selected components to a new district
        #if the component is the whole district, skip it as it wont produce a valid districting
        for selected_component in selected_components:
            original_district_id = dist_dict[next(iter(selected_component))]  # Get the district ID of the first node in the component
            nodes_in_original_district = {node for node in dist_dict if dist_dict[node] == original_district_id}
            if selected_component == nodes_in_original_district:
                continue

            # Check if the component is connected in the original district after removing the selected edges
            new_curr_district_nodes = {node for node in graph.nodes if dist_dict_proposal.get(node) == original_district_id} - selected_component
            if not new_curr_district_nodes or not nx.is_connected(graph.subgraph(new_curr_district_nodes)):
                continue

            #get the district to move the component to
            connected_districts = {dist_dict.get(neighbor) for node in selected_component for neighbor in graph.neighbors(node)} - {original_district_id}
            if not connected_districts:
                continue

            new_district_id = random.choice(list(connected_districts))
            component_population = sum(graph.nodes[node]['voters'] for node in selected_component)

            #move the component to the proposed district
            for node in selected_component:
                dist_dict_proposal.update({node: new_district_id})
            dist_population_dict_proposal[new_district_id] += component_population
            dist_population_dict_proposal[original_district_id] -= component_population

            if step > hot_steps:
                # Set the needs_update flag for the old and new districts
                dist_gdf_proposal.loc[original_district_id, 'needs_update'] = True
                dist_gdf_proposal.loc[new_district_id, 'needs_update'] = True


        # If step is a hot step, accept the proposal without checking the acceptance probability or counting the energy
        if step < hot_steps:
            dist_dict = dist_dict_proposal
            dist_population_dict = dist_population_dict_proposal

            not_active_edges = {(u, v) for u, v in graph.edges() if dist_dict.get(u) != dist_dict.get(v)}
            active_edges = set(graph.edges) - not_active_edges
            boundary_nodes = {node for edge in not_active_edges for node in edge}

            accepted_count += 1
        
        # The last step of the hot phase - we need to start calculating parameters for the annealing phase
        elif step == hot_steps:
            dist_dict = dist_dict_proposal
            dist_population_dict = dist_population_dict_proposal

            not_active_edges = {(u, v) for u, v in graph.edges() if dist_dict.get(u) != dist_dict.get(v)}
            active_edges = set(graph.edges) - not_active_edges
            boundary_nodes = {node for edge in not_active_edges for node in edge}
            
            districts = [[node for node in dist_dict if dist_dict.get(node) == i] for i in range(number_of_districts)]
            dist_gdf = gpd.GeoDataFrame({
                'id': range(number_of_districts),
                'geometry': [polygon_gdf[polygon_gdf['graph_node_id'].isin(dist)].union_all() for dist in districts],
                'compactness': [0] * number_of_districts,
                'needs_update': False,
            })

            # Calculate compactness for each district - area / area of (the convex hull intersected with the border)
            dist_gdf['compactness'] = dist_gdf.apply(
                lambda row: row['geometry'].area / (row['geometry'].convex_hull.intersection(border_geometry).area),
                axis=1
            )

            compactness = dist_gdf['compactness'].mean()
            pop_max_deriv = max(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1))
            pop_avg_deriv = sum(np.abs((np.array(list(dist_population_dict.values())) / avg_voters) - 1) ** 2)

            #print(f"Hot step {step}/{hot_steps + annealing_steps + cold_steps} - starting to calculate acceptance probability")
            accepted_count += 1
        else:
            pop_max_deriv_proposal = max(np.abs((np.array(list(dist_population_dict_proposal.values())) / avg_voters) - 1))
            pop_avg_deriv_proposal = sum(np.abs((np.array(list(dist_population_dict_proposal.values())) / avg_voters) - 1) ** 2)

            # Update the district GeoDataFrame in rows that have flag 'needs_update' set to True
            needs_update_ids = dist_gdf_proposal.index[dist_gdf_proposal['needs_update']]
            for district_id in needs_update_ids:
                dist_nodes = [node for node, d_id in dist_dict_proposal.items() if d_id == district_id]
                geom = polygon_gdf[polygon_gdf['graph_node_id'].isin(dist_nodes)].union_all()
                dist_gdf_proposal.at[district_id, 'geometry'] = geom
                # Compactness: area / (convex hull intersected with border)
                border_area = geom.convex_hull.intersection(border_geometry).area
                dist_gdf_proposal.at[district_id, 'compactness'] = geom.area / (border_area if border_area > 0 else 1)
            dist_gdf_proposal['needs_update'] = False  # Reset the needs_update flag

            compactness_proposal = dist_gdf_proposal['compactness'].sum()  # Average compactness of districts

            energy_new = pop_eq_beta * pop_avg_deriv_proposal + comp_beta * (1 - compactness_proposal)  # Calculate new energy
            energy_old = pop_eq_beta * pop_avg_deriv + comp_beta * (1 - compactness)

            acceptance_probability = exp(-max(0, energy_new - energy_old))  # Calculate acceptance probability

            # print(f"Avg Derivation Proposal: {pop_avg_deriv_proposal}, Avg Derivation: {pop_avg_deriv}, Compactness Proposal: {compactness_proposal}, Compactness: {compactness}, Acceptance Probability: {acceptance_probability}")
            
            if random.random() <= acceptance_probability:
                dist_dict = dist_dict_proposal
                dist_population_dict = dist_population_dict_proposal
                dist_gdf = dist_gdf_proposal.copy()

                not_active_edges = {(u, v) for u, v in graph.edges() if dist_dict.get(u) != dist_dict.get(v)}
                active_edges = set(graph.edges) - not_active_edges
                boundary_nodes = {node for edge in not_active_edges for node in edge}
                compactness = compactness_proposal
                pop_max_deriv = pop_max_deriv_proposal
                pop_avg_deriv = pop_avg_deriv_proposal

                #print(f"Accepted at step {step}/{annealing_steps + cold_steps + hot_steps}, p = {acceptance_probability}")
                accepted_count += 1

            else:
                # For information purposes only
                #print(f"Rejected at step {step}/{annealing_steps + cold_steps + hot_steps}, p = {acceptance_probability}")
                rejected_count += 1

        #update the parameters
        if step > hot_steps and step <= hot_steps + annealing_steps:
            pop_eq_beta += beta_eq_pop_target / annealing_steps 
            comp_beta += beta_compactness_target / annealing_steps

        if step % 100 == 0:
            print(f"Step {step}/{annealing_steps + cold_steps + hot_steps}\nAvg Derivation: {pop_avg_deriv}, Compactness: {compactness}\nAccepted: {accepted_count}, Rejected: {rejected_count}\npop_target: {pop_eq_beta}, comp_target: {comp_beta}")
            try:
                progress_data[int(step / 100), 0] = accepted_count
                progress_data[int(step / 100), 1] = rejected_count
                progress_data[int(step / 100), 2] = pop_max_deriv
                progress_data[int(step / 100), 3] = pop_avg_deriv
                progress_data[int(step / 100), 4] = compactness
            except IndexError:
                pass
            accepted_count = 0
            rejected_count = 0

        step += 1

    if len(set(dist_dict.values())) != number_of_districts:
        raise ValueError("Unexpected error in redistricting - number of districts is not correct")
    
    #print final results
    print(f"Final Avg Derivation: {pop_avg_deriv}, Final Compactness: {compactness}, Final Max Derivation: {pop_max_deriv}\n beta_eq_pop_target: {pop_eq_beta}, beta_compactness_target: {comp_beta}")

    return [[node for node in dist_dict if dist_dict.get(node) == i] for i in range(number_of_districts)], progress_data

def favouritism_alg(graph: nx.Graph,
                    number_of_districts: int,
                    parties_names: list,
                    party_to_favour: str,
                    initial_seeding_attempts: int = 500,
                    deviation: float = 0.01,
                    delta_percentage_weight: float = 0.2,
                    convergence_acceptance: float = 0.05,
                    seed: int = int(time.time())) -> list:

    """
    Favouritism algorithm for redistricting. Alghorithm creates districts with the goal of maximizing the number of districts where the party to favour has the most votes.

    Parameters
    ----------
    graph : networkx.Graph
        A graph representing the geographical units.
        Each node should have a 'voters' attribute as well as number of votes for each party defined in parties_names. Name of each atribute should be the same as the party name in parties_names.

    number_of_districts : int
        Number of districts to form.

    parties_names : list
        List of party names. Each party should have a corresponding attribute in the graph.

    party_to_favour : str
        Name of the party to favour, the same as in parties_names.

    initial_seeding_attempts : int, optional
        Number of initial seeding attempts (guesses). Default is 500.

    deviation : float, optional
        The deviation from the average number of voters in a district.
        It is used for the initial guess - seeding algorithm. Final deviation can be different.
        For 5% deviation, use 0.05. Default is 0.01 (1%).
        Derivation has to be between 0 and 1 (0% and 100%).

    delta_percentage_weight : float, optional
        The weight of the percentage difference in votes for the party to favour.
        (If the percentage difference is greater than this value, the algorithm will accept the move, regardless of the change in the number of wins)

    convergence_acceptance : float, optional
        The acceptance rate for the algorithm to converge. Default is 0.05 (5%).
        Per 200 steps, if the acceptance rate is below this value and the number of wins is not increasing, the algorithm stops.
        Acceptance rate has to be between 0 and 1 (0% and 100%).

    seed : int, optional
        Random seed. Default is the current time.

    Returns
    -------
    list
        List of districts, where each district is a list of node IDs.
    """

    random.seed(seed)

    # Get index of the party to favour
    party_index = parties_names.index(party_to_favour)

    # Seeding and Initialization - Find the best initial seeding guess
    best_wins = -1
    closest_votes_to_win = np.inf
    districts = []
    for i in range(initial_seeding_attempts):
        temp = seeding_algorithm(graph, number_of_districts, deviation, seed + i)
        wins = 0
        votes_to_win = np.inf

        for dist in temp:
            votes = [0] * len(parties_names)
            for node in dist:
                for j, party in enumerate(parties_names):
                    votes[j] += graph.nodes[node][party]
            if max(votes) == votes[party_index]:
                wins += 1
            else:
                votes_to_win = min(votes_to_win, max(votes) - votes[party_index])

        if wins > best_wins:
            best_wins = wins
            districts = temp
        if wins == best_wins and votes_to_win < closest_votes_to_win:
            closest_votes_to_win = votes_to_win
            districts = temp
            
    # Prepare data structures
    dist_dict = {} # Node to district mapping
    dist_population_dict = {} # District to population mapping 
    total_population = sum(graph.nodes[node].get('voters', 0) for node in graph.nodes)
    avg_voters = int(total_population / number_of_districts)

    for district_index, dist in enumerate(districts):
        for node in dist:
            dist_dict[node] = district_index
            dist_population_dict[district_index] = dist_population_dict.get(district_index, 0) + graph.nodes[node]['voters']
    
    best_dist_dict = dist_dict.copy() # Best districting (at the moment)

    # Mark inactive edges
    not_active_edges = set()
    for edge in graph.edges:
        if dist_dict.get(edge[0]) != dist_dict.get(edge[1]):
            not_active_edges.add(edge)
            graph.edges[edge]['is_active'] = False
        else:
            graph.edges[edge]['is_active'] = True

    # Metrics Initialization
    voters_deviation = sqrt(sum((avg_voters - v)**2 for v in dist_population_dict.values()) / avg_voters)
    tau = 0.01  # Initial temperature
    step = 1
    accepted = 0
    accepted_sum = 0
    wins = best_wins

    # Main Optimization Loop
    while True:
        if wins > best_wins:
            best_wins = wins
            best_dist_dict = dist_dict.copy()
            
        if step % 200 == 0:
            acceptance_rate = accepted / 200
            accepted_sum += acceptance_rate
            accepted = 0

            # Convergence check
            if acceptance_rate <= convergence_acceptance and wins == best_wins:
                best_dist_dict = dist_dict.copy()
                break
            if step % 10000 == 0:
                print(f"Acceptance rate: {acceptance_rate:.3f}, Step: {step}")
                print(f"Best wins: {best_wins}, Current wins: {wins}")
                if accepted_sum / (10000/200) <= 0.5*convergence_acceptance:
                    break
                accepted_sum = 0
                tau *= 0.85  # Gradual cooling

        # Choose a random inactive edge and nodes
        edge = random.choice(list(not_active_edges))
        selected_node = random.choice(edge)
        other_node = edge[0] if edge[1] == selected_node else edge[1]

        # Current and new districts
        curr_district = dist_dict[selected_node]
        new_district = dist_dict[other_node]

        # Validate district contiguity after removal
        new_curr_district_nodes = {node for node in dist_dict if dist_dict[node] == curr_district} - {selected_node}
        if not new_curr_district_nodes or not is_contiguous(new_curr_district_nodes, graph):
            step += 1
            continue

        # Compute new deviation and wins
        new_voters_deviation = compute_voters_deviation(
            voters_deviation, dist_population_dict, avg_voters, selected_node, curr_district, new_district, graph)

        delta_wins, delta_percentage = compute_delta_wins(
            curr_district, new_district, selected_node, party_index, dist_dict, graph, parties_names)

        # Acceptance probability
        if delta_wins > 0:
            acceptance_probability = 1.0 # Always accept if the party to favour wins
        elif delta_percentage > delta_percentage_weight:
            acceptance_probability = exp(delta_percentage) * exp(delta_wins * 10)
        else:
            delta_deviation = new_voters_deviation - voters_deviation
            try:
                acceptance_probability = exp(-delta_deviation / tau) * exp(delta_wins * 1/(2*tau))
            except OverflowError:
                if -delta_deviation < 0:
                    acceptance_probability = 0
                else:
                    acceptance_probability = 1

        # Accept or Reject the move
        if random.random() < acceptance_probability:
            dist_dict[selected_node] = new_district
            dist_population_dict[curr_district] -= graph.nodes[selected_node]['voters']
            dist_population_dict[new_district] += graph.nodes[selected_node]['voters']
            voters_deviation = new_voters_deviation
            wins += delta_wins
            accepted += 1

            # Update edge activity
            update_edge_activity(not_active_edges, graph, dist_dict)

        step += 1

    # Generate final districts
    return [[node for node in dist_dict if best_dist_dict[node] == i] for i in range(number_of_districts)]

def is_contiguous(nodes, graph):
    """Check if a set of nodes is contiguous."""
    visited = set()
    stack = [next(iter(nodes))]
    while stack:
        current = stack.pop()
        visited.add(current)
        stack.extend(neighbor for neighbor in graph.neighbors(current) if neighbor in nodes and neighbor not in visited)
    return visited == nodes

def compute_voters_deviation(voters_deviation, dist_population_dict, avg_voters, node, curr_district, new_district, graph):
    """Compute the new voters deviation."""
    curr_pop = dist_population_dict[curr_district]
    new_pop = dist_population_dict[new_district]
    voters = graph.nodes[node]['voters']

    return sqrt(
        ((voters_deviation ** 2 * avg_voters)
         - (curr_pop - avg_voters) ** 2 - (new_pop - avg_voters) ** 2
         + (curr_pop - voters - avg_voters) ** 2
         + (new_pop + voters - avg_voters) ** 2) / avg_voters)

def compute_delta_wins(curr_district, new_district, node, party_index, dist_dict, graph, parties_names):
    """Compute the change in wins and weighted change in votes for the favored party."""
    curr_votes = compute_votes(curr_district, dist_dict, graph, parties_names)
    new_votes = compute_votes(new_district, dist_dict, graph, parties_names)
    proposed_curr_votes = curr_votes.copy()
    proposed_new_votes = new_votes.copy()

    for j, party in enumerate(parties_names):
        votes = graph.nodes[node][party]
        proposed_curr_votes[j] -= votes
        proposed_new_votes[j] += votes

    # Compute change in wins
    delta_wins = 0
    if curr_votes[party_index] == max(curr_votes):
        delta_wins -= 1
    if new_votes[party_index] == max(new_votes):
        delta_wins -= 1
    if proposed_curr_votes[party_index] == max(proposed_curr_votes):
        delta_wins += 1
    if proposed_new_votes[party_index] == max(proposed_new_votes):
        delta_wins += 1

    # Compute change in percentage support
    total_curr_votes = sum(proposed_curr_votes)
    total_new_votes = sum(proposed_new_votes)

    curr_party_percentage = (proposed_curr_votes[party_index] / total_curr_votes) if total_curr_votes > 0 else 0
    new_party_percentage = (proposed_new_votes[party_index] / total_new_votes) if total_new_votes > 0 else 0

    delta_percentage = new_party_percentage - curr_party_percentage

    # Return both changes
    return delta_wins, delta_percentage

def compute_votes(district, dist_dict, graph, parties_names):
    """Compute the total votes in a district."""
    votes = [0] * len(parties_names)
    for node in dist_dict:
        if dist_dict[node] == district:
            for j, party in enumerate(parties_names):
                votes[j] += graph.nodes[node][party]
    return votes

def update_edge_activity(not_active_edges, graph, dist_dict):
    """Update the active/inactive status of edges."""
    not_active_edges.clear()
    for edge in graph.edges:
        if dist_dict[edge[0]] != dist_dict[edge[1]]:
            not_active_edges.add(edge)
            graph.edges[edge]['is_active'] = False
        else:
            graph.edges[edge]['is_active'] = True

def graph_coarsening(graph: nx.Graph, decrease_percent: int = 0.5, num_of_iterations: int = 1) -> list:
    """
    Coarsen the graph by merging nodes based on the specified factor.
    
    Parameters
    ----------
    graph : networkx.Graph
        The input graph to be coarsened.
    
    decrease_percent : float, optional
        The percentage by which the number of nodes in the graph should be reduced.
        Default is 0.5 (50% reduction - half of the original number of nodes).

    num_of_iterations : int, optional
        The number of iterations to perform the coarsening.
        Default is 1 iterations - return 1 graph with 50% of the original number of nodes each time.
    
    Returns
    -------
    list
        A list of coarsened graphs, each with approximately the specified percentage of nodes reduced.
    """
    
    # Validate input parameters
    # TODO

    coarsened_graphs = []
    main_graph = graph.copy()

    # Assigning a weight to each edge based on the number of voters in the nodes it connects
    for e in main_graph.edges:
        main_graph.edges[e]['weight'] = (main_graph.nodes[e[0]].get('voters', 0) + main_graph.nodes[e[1]].get('voters', 0))
    
    for _ in range(num_of_iterations):
        print(f"Coarsening iteration {_ + 1}/{num_of_iterations}")
        
        _graph = main_graph.copy()

        #Normalize the weights by dividing by the minimum weight (to avoid counting e^(-very_high_weight))
        min_weight = min(_graph.edges[e]['weight'] for e in _graph.edges)
        for e in _graph.edges:
            _graph.edges[e]['exp'] = exp(-(_graph.edges[e]['weight'] / min_weight))

        while len(_graph.nodes) > int(len(graph.nodes) * (1 - decrease_percent)):
            # Select a random edge based on the wieghts
            edges = list(_graph.edges)
            edge_weights = [_graph.edges[e]['exp'] for e in _graph.edges]

            selected_edges = random.choices(edges, weights=edge_weights, k=50)

            for selected_edge in selected_edges:
                # If the selected edge is not in the graph already, skip it
                if selected_edge not in _graph.edges:
                    continue

                if selected_edge in main_graph.edges:
                    main_graph.edges[selected_edge]['weight'] *= 2 # Double the weight of the edge in the main graph to avoid selecting it again

                # Merge the nodes connected by the selected edge
                node1, node2 = selected_edge
                new_node = f"{node1}+{node2}"
                
                # Create a new node with combined attributes
                _graph.add_node(new_node, voters=_graph.nodes[node1].get('voters', 0) + _graph.nodes[node2].get('voters', 0))
                
                # Update edges to point to the new node
                for neighbor in list(_graph.neighbors(node1)) + list(_graph.neighbors(node2)):
                    if neighbor != node1 and neighbor != node2:
                        _graph.add_edge(new_node, neighbor, weight=_graph.nodes[node1].get('voters', 0) + _graph.nodes[node2].get('voters', 0) + _graph.nodes[neighbor].get('voters', 0))
                
                # Remove old nodes and edges
                _graph.remove_node(node1)
                _graph.remove_node(node2)

            new_min_weight = min(_graph.edges[e]['weight'] for e in _graph.edges)
            if new_min_weight != min_weight:
                # Recalculate the minimum weight and exp values if the minimum weight has changed
                min_weight = new_min_weight
                for e in _graph.edges:
                    _graph.edges[e]['exp'] = exp(-(_graph.edges[e]['weight'] / new_min_weight))
            else:
                # If the minimum weight has not changed, give exp values only to new edges
                for e in _graph.edges:
                    if 'exp' not in _graph.edges[e]:
                        _graph.edges[e]['exp'] = exp(-(_graph.edges[e]['weight'] / min_weight))

            if int((len(graph.nodes) - len(_graph.nodes)) / (len(graph.nodes) * decrease_percent) * 100) % 5 == 0:
                print("Coarsening progress: ", (len(graph.nodes) - len(_graph.nodes)) / (len(graph.nodes) * decrease_percent) * 100, "%")

        # Append the coarsened graph to the list
        coarsened_graphs.append(_graph)

    # For every graph, remove the 'exp' and 'weight' attribute from edges
    for g in coarsened_graphs:
        for e in g.edges:
            if 'exp' in g.edges[e]:
                del g.edges[e]['exp']
            if 'weight' in g.edges[e]:
                del g.edges[e]['weight']

    # Return the list of coarsened graphs
    return coarsened_graphs

def run_redist_flip_alg_pop(i):
        print(f"Running redist_flip_alg with thread {i + 1}/{N_OF_THREADS}")
        new, old, data = redist_flip_alg(
            graph, gdf_dissolved, poland, 100,
            hot_steps=500, annealing_steps=200, cold_steps=20000,
            lambda_prob=0.1, beta_eq_pop_target=2000, beta_compactness_target=0,
            seed=i + int(time.time())
        )
        print(f"Finished redist_flip_alg with thread {i + 1}/{N_OF_THREADS}")
        pkl.dump(new, open(f"temp/redist_flip_alg_pop_{i}.pkl", "wb"))
        pkl.dump(data, open(f"temp/redist_flip_alg_data_pop_{i}.pkl", "wb"))

def run_redist_flip_alg_comp(i):
        print(f"Running redist_flip_alg with thread {i + 1}/{N_OF_THREADS}")
        new, old, data = redist_flip_alg(
            graph, gdf_dissolved, poland, 100,
            hot_steps=500, annealing_steps=200, cold_steps=30000,
            lambda_prob=0.1, beta_eq_pop_target=0, beta_compactness_target=800,
            seed=i + int(time.time())
        )
        print(f"Finished redist_flip_alg with thread {i + 1}/{N_OF_THREADS}")
        pkl.dump(new, open(f"temp/redist_flip_alg_comp_{i}.pkl", "wb"))
        pkl.dump(data, open(f"temp/redist_flip_alg_data_comp_{i}.pkl", "wb"))

def run_redist_flip_alg_test(beta_eq_pop_target=2000, beta_compactness_target=4000, i=0):
        print(f"Running redist_flip_alg with thread {i + 1}/{N_OF_THREADS}")
        new, old, data = redist_flip_alg(
            graph, gdf_dissolved, poland, 100,
            hot_steps=10, annealing_steps=100, cold_steps=5000,
            lambda_prob=0.1, beta_eq_pop_target=beta_eq_pop_target, beta_compactness_target=beta_compactness_target
        )
        print(f"Finished redist_flip_alg with thread {i}/{N_OF_THREADS}")
        pkl.dump(new, open(f"temp/redist_flip_alg_{beta_eq_pop_target}_{beta_compactness_target}.pkl", "wb"))
        pkl.dump(data, open(f"temp/redist_flip_alg_data_{beta_eq_pop_target}_{beta_compactness_target}.pkl", "wb"))

N_OF_THREADS = 5
graph = pkl.load(open("temp/coarsened_graphs.pkl", "rb"))[0]
poland = gpd.read_file("shapefiles/polska/polska.shp", encoding="utf-8")
gdf_dissolved = pkl.load(open("temp/gdf_dissolved.pkl", "rb"))

if __name__ == '__main__':
# graph = nx.read_gexf("pickle_files/wybory2019sejm_graph_wth_parties.gexf")
# print(len(graph.edges))
# #change name of property Wyborcy to voters
# for node in graph.nodes:
#     if 'wyborcy' in graph.nodes[node]:
#         graph.nodes[node]['voters'] = graph.nodes[node]['wyborcy']
#         del graph.nodes[node]['wyborcy']

# output = graph_coarsening(graph)

# pkl.dump(output, open("temp/coarsened_graphs.pkl", "wb"))

    
# gdf = gpd.read_file("shapefiles/wybory2019_voronoi/wybory2019sejm_voronoi.shx", encoding="utf-8")
# gdf = gdf[['teryt', 'obwod', 'geometry']]
# gdf['id'] = gdf['teryt'] + '_' + gdf['obwod'].astype(str)

# simple_id_to_graph_node_map = {}
# for graph_node in graph.nodes:
#     simple_ids = graph_node.split('+')
#     for simple_id in simple_ids:
#         # Ponieważ wiemy, że nie ma konfliktów, możemy bezpiecznie przypisywać wartości.
#         simple_id_to_graph_node_map[simple_id] = graph_node

# gdf['graph_node_id'] = gdf['id'].map(simple_id_to_graph_node_map)

# # Opcjonalnie: sprawdź, czy któreś wiersze nie zostały zmapowane
# unmapped_rows = gdf[gdf['graph_node_id'].isnull()]
# if not unmapped_rows.empty:
#     print(f"\nUWAGA: Nie znaleziono mapowania dla {len(unmapped_rows)} wierszy. ID, których nie było w grafie:")
#     print(unmapped_rows['id'].unique())
# # Wykonujemy dissolve na nowej kolumnie z ID węzłów grafu.
# gdf_dissolved = gdf.dissolve(by='graph_node_id', as_index=False)

# pkl.dump(gdf_dissolved, open("temp/gdf_dissolved.pkl", "wb"))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_OF_THREADS) as executor:
            futures = [executor.submit(run_redist_flip_alg_test, 3100, 3020, 1), executor.submit(run_redist_flip_alg_test, 3000, 2700, 2), executor.submit(run_redist_flip_alg_test, 3000, 3000, 3), executor.submit(run_redist_flip_alg_test, 2700, 3000, 4), executor.submit(run_redist_flip_alg_test, 4000, 2000, 5)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'Proces wygenerował wyjątek: {exc}')

    #new, old, data = redist_flip_alg(graph, gdf_dissolved, poland, 100, hot_steps=5, annealing_steps=5, cold_steps=1000, lambda_prob=0.1, beta_eq_pop_target=2000, beta_compactness_target=4000, seed=42)
    # pkl.dump(new, open("temp/redist_flip_alg.pkl", "wb"))
    # pkl.dump(data, open("temp/redist_flip_alg_data.pkl", "wb"))


# merged_gdf = gpd.read_file("temp/gminy_sejm_2023/gminy_sejm_2023.shp", encoding="utf-8")
# party_list = ["KOMIT_1", "KOMIT_2", "KOMIT_3", "KOMIT_4", "KOMIT_5", "KOMIT_6", "KOMIT_7", "KOMIT_8", "KOMIT_9", "KOMIT_10", "KOMIT_11", "KOMIT_12"]
# #add votes to the graph
# for index, row in merged_gdf.iterrows():
#     for party in party_list:
#         # if party in ["KOMIT_2", "KOMIT_3", "KOMIT_6"]:
#         #     graph.nodes[row["JPT_KOD_JE"]]["ZO"] = graph.nodes[row["JPT_KOD_JE"]].get("ZO", 0) + row[party]
#         # else:
#         graph.nodes[row["JPT_KOD_JE"]][party] = row[party]

# for favoured_party in ["KOMIT_4"]:
#     print(favoured_party)
#     new, data = favouritism_alg(graph, 100, party_list, favoured_party, 0.01, 0.01)
#     pkl.dump(new, open(f"temp/favouritism_alg_{favoured_party}.pkl", "wb"))
#     pkl.dump(data, open(f"temp/favouritism_alg_data_{favoured_party}.pkl", "wb"))

# # new,data = favouritism_alg(graph, 100, party_list, favoured_party, 0.01, 0.01)
# # pkl.dump(new, open("temp/favouritism_alg.pkl", "wb"))
# # pkl.dump(data, open("temp/favouritism_alg_data.pkl", "wb"))

# # # new, old = graph_cut_algorithm(graph, 100, max_iterations=2000, T=200, num_of_chains=2, lambda_val=5, beta_end=3)

# # threads = []
# # results = [None] * 2

# # def run_algorithm(index, *args):
# #     results[index] = redist_flip_alg(*args)

# # threads.append(threading.Thread(target=run_algorithm, args=(0, graph, 20, 10, 2000, 200, 0.05, 0.1, 100, 1)))
# # #threads.append(threading.Thread(target=run_algorithm, args=(1, graph, 20, 100, 2000, 200, 0.05, 1, 0.1, 1)))

# # for thread in threads:
# #     thread.start()

# # for thread in threads:
# #     thread.join()

# # new1, old1, data1 = results[0]
# # # new2, old2, data2 = results[1]

# # pkl.dump(new1, open("temp/graph_cut_algorithm_1.pkl", "wb"))
# # pkl.dump(old1, open("temp/graph_cut_algorithm_start_1.pkl", "wb"))
# # pkl.dump(data1, open("temp/redist_flip_alg_data_1.pkl", "wb"))

# # pkl.dump(new2, open("temp/graph_cut_algorithm_2.pkl", "wb"))
# # pkl.dump(old2, open("temp/graph_cut_algorithm_start_2.pkl", "wb"))
# # pkl.dump(data2, open("temp/redist_flip_alg_data_2.pkl", "wb"))



# # new, old, data = redist_flip_alg(graph, 100, hot_steps=500, annealing_steps=2000, cold_steps=1000, lambda_prob=0.1, pop_tol_target=0.95, comp_weight_target=0.05)

# # pkl.dump(new, open("temp/graph_cut_algorithm.pkl", "wb"))
# # pkl.dump(old, open("temp/graph_cut_algorithm_start.pkl", "wb"))
# # pkl.dump(data, open("temp/redist_flip_alg_data.pkl", "wb"))