from copy import deepcopy
from math import exp
import math
import networkx as nx
import numpy as np
import random
import time
import threading

def seeding_algorithm(graph: nx.Graph,
                      number_of_districts: int,
                      deriviation: float = 0.01,
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

    deriviation : float, optional
        The deviation from the average number of voters in a district. 
        For 5% deviation, use 0.05. Default is 0.01 (1%).
        Derivation has to be between 0 and 1 (0% and 100%).

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

    if deriviation > 1 or deriviation < 0:
            raise ValueError("Derivation has to be between 0 and 1")

    def seeding_algorithm_inner(graph: nx.Graph,
                                number_of_districts: int,
                                deriviation: float = 0.01,
                                seed: int = int(time.time()),
                                ) -> list:
        """
        Helper function for seeding_algorithm (to prevent isolation of nodes - if occurs, then rerun the seeding algorithm)
        """
        graph_copy = deepcopy(graph)

        #Calculate the average number of voters in a district
        total_voters = 0
        for node in graph_copy.nodes:
            try:
                total_voters += graph_copy.nodes[node]['voters']
            except KeyError:
                raise ValueError("Each node in the graph has to have a 'voters' attribute")
        
        try:
            average_voters = int(total_voters / number_of_districts)
        except ZeroDivisionError:
            raise ValueError("Number of districts has to be greater than 0")

        #Every county with more than the average number of voters (+/- deriviation) is considered a district and is removed from the graph
        districts = []
        for node in graph_copy.nodes:
            if graph_copy.nodes[node]['voters'] > average_voters * (1 + abs(deriviation)):
                districts.append([node])
        for district in districts:
            graph_copy.remove_node(district[0])

        number_of_districts -= len(districts)
        random.seed(seed)
        #Creating the remaining districts as long as there are nodes in the graph
        while len(graph_copy.nodes) > number_of_districts and len(graph_copy.nodes) > 0 and number_of_districts > 0:
            district = []
            #Start district from a random county
            district.append(random.choice(list(graph_copy.nodes)))
            current_voters = graph_copy.nodes[district[0]]['voters']

            #Add neighbors to the district until the number of voters is close to the average
            while current_voters < average_voters * (1 - deriviation) and len(graph_copy.nodes) > number_of_districts:
                #Add all neighbors of the district
                neighbors = set()
                for node in district:
                    neighbors.update(list(graph_copy.neighbors(node)))
                
                #Remove nodes that are already in the district
                neighbors = neighbors - set(district)

                #District is as good as it gets
                if len(neighbors) == 0:
                    break

                #Remove nodes that have too many voters
                neighbors_copy = neighbors.copy()
                for neighbor in neighbors_copy:
                    if graph_copy.nodes[neighbor]['voters'] + current_voters > average_voters * (1 + deriviation):
                        neighbors.remove(neighbor)

                #If there are no neighbors left, add the one that is the closest to the average number of voters
                #if the average number of voters is closer to the current number of voters with the best neighbor
                if len(neighbors) == 0:
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
                
                #Try to find a neighbor the most compact to the rest of the district
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

        if len(missing_nodes) > 0.05 * len(graph.nodes):
            return [], True
        
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
                        raise ValueError("Unexpected error in seeding algorithm")

            missing_nodes = set(graph.nodes) - covered_nodes
            if prev_len_missing_nodes == len(missing_nodes):
                raise ValueError("Unexpected error in seeding algorithm")

        #If there are districts with only one node, add them to the list of districts
        if number_of_districts > 0:
            for node in graph_copy.nodes:
                districts.append([node])

        return districts, False
 
    i = 100
    while i > 0:
        try:
            result, try_again = seeding_algorithm_inner(graph, number_of_districts, deriviation, seed)
            if try_again:
                deriviation *= 1.5
                i -= 1
            else:
                return result
        except ValueError as e:
            i -= 1
            print(f"Error in seeding algorithm - retrying {100 - i}/100")
            seed += 1

    raise ValueError("Unexpected error in seeding algorithm")

def graph_cut_algorithm(graph: nx.Graph,
                        number_of_districts: int,
                        deriviation: float = 0.01,
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

    deriviation : float, optional
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
            districts = seeding_algorithm(graph, number_of_districts, deriviation, seed)
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

def redist_flip_alg(graph: nx.Graph,
                    number_of_districts: int,
                    hot_steps: int = 500,
                    annealing_steps: int = 1000,
                    cold_steps: int = 100,
                    lambda_prob: float = 0.01,
                    pop_tol_target: float = 0.95,
                    comp_weight_target: float = 0.5,
                    seed: int = int(time.time()),
                    ) -> list:
    """
    Redistricting algorithm by flipping edges, using simulated annealing.
    
    Parameters
    ----------
    graph : networkx.Graph
        A graph representing the counties and their neighbors.
        Each node in the graph needs to have a 'voters' attribute representing the number of voters in the county.

    number_of_districts : int
        The number of districts to create.

    hot_steps : int, optional
        The number of steps at the beginning of the algorithm where the acceptance rate is 100%. Default is 500 steps.

    annealing_steps : int, optional
        The number of steps where the temperature is decreasing. Default is 1000 steps.

    cold_steps : int, optional
        The number of steps at the end of the algorithm where the temperature is 0. Default is 100 steps.

    lambda_prob : float, optional
        The probability of flipping an edge. Default is 5%.
        Value has to be between 0 and 1.
    
    pop_tol_target : float, optional
        The target population tolerance. Default is 0.1 (10%).
        Population tolerance means the maximum deviation from the average population in a district.
        Value has to be between 0 and 1.

    comp_weight_target : float, optional
        The target compactness weight. Default is 0.5.
        Compactness weight is the ratio of the number of active edges to the total number of edges in the graph to ensure compact districts.

    seed : int, optional
        The seed for the random number generator. Default is the current time.

    Returns
    -------
    list
        Array of districts. Each district is a list of counties' IDs (acoording to the graph).
        For a district with one county, the list will contain one element.

    References
    ----------
    [1] https://doi.org/10.1016/S0962-6298(99)00047-5
    """
    
    if lambda_prob < 0 or lambda_prob > 1:
        raise ValueError("Lambda probability has to be between 0 and 1")
    if number_of_districts < 1:
        raise ValueError("Number of districts has to be greater than 0")
    if pop_tol_target < 0 or pop_tol_target > 1:
        raise ValueError("Population tolerance target has to be between 0 and 1")
    if comp_weight_target < 0 or comp_weight_target > 1:
        # raise ValueError("Compactness weight target has to be between 0 and 1") TODO: Check if this is correct
        pass
    if hot_steps < 1:
        raise Warning("Hot steps should be greater than 0, but if you want to skip the hot steps, you're free to do so")
        hot_steps = 0
    if cold_steps < 1:
        raise Warning("Cold steps should be greater than 0, but if you want to skip the hot steps, you're free to do so")
        cold_steps = 0
    if annealing_steps < 1:
        raise ValueError("Annealing steps should be greater than 0")
        annealing_steps = 0

    districts = seeding_algorithm(graph, number_of_districts, seed=seed)
    dist_dict = {}
    dist_population_dict = {}
    dist_graph = nx.Graph(graph)
    try:
        total_population = sum(graph.nodes[node]['voters'] for node in graph.nodes)
        avg_voters = total_population / number_of_districts
    except KeyError:
        raise ValueError("Each node in the graph has to have a 'voters' attribute coreesponding to the number of voters in the county")

    pop_eq = 0
    district_index = 0
    for dist in districts:
        for node in dist:
            dist_dict.update({node: district_index})
            dist_population_dict.update({district_index: dist_population_dict.get(district_index, 0) + graph.nodes[node]['voters']})
            for neighbor in graph.neighbors(node):
                if neighbor in dist:
                    dist_graph.edges[node, neighbor]['is_active'] = True
        pop_eq += abs(dist_population_dict.get(district_index, 0) - avg_voters)
        district_index += 1
    
    # pop_eq /= avg_voters

    #Getting boundary nodes - v such that N(v) & S_δ(S_delta) is not empty
    not_active_edges = set()
    boundary_nodes = set()
    for edge in dist_graph.edges:
        if 'is_active' not in dist_graph.edges[edge]:
            dist_graph.edges[edge]['is_active'] = False
            not_active_edges.add(edge)
            boundary_nodes.add(edge[0])
            boundary_nodes.add(edge[1])

    active_edges = set(graph.edges) - not_active_edges
    
    comp = len(not_active_edges) / len(dist_graph.edges)
    print(f"Initial compactness: {comp}")

    district_proposal = deepcopy(dist_graph)
    dist_dict_proposal = dist_dict.copy()
    dist_population_dict_proposal = dist_population_dict.copy()
    pop_eq_proposal = pop_eq
    prev_exponent = 0
    pop_eq_beta, comp_beta, steps = 0, 0, 0

    random.seed(seed)
    rejected = 0
    accepted = 0
    data = np.zeros((int((annealing_steps + cold_steps + hot_steps)/100), 2))
    while steps < annealing_steps + hot_steps + cold_steps:
        #set flipped edges
        flipped_edges = {edge for edge in active_edges if random.random() <= lambda_prob}
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(graph.nodes)
        temp_graph.add_edges_from(flipped_edges)

        # Identify boundary-connected components
        boundary_components = [comp for comp in nx.connected_components(temp_graph) if any(node in boundary_nodes for node in comp)]
            
        if len(boundary_components) == 0:
            continue

        #select a subset of boundary components not connecting to each other
        R = max(min(len(boundary_components) - 1, np.random.normal(0.2*len(boundary_components), 0.2*len(boundary_components))), 1)
        selected_components, selected_nodes = [], set()
        unchanged_counter = 0
        while len(selected_components) < R and boundary_components:
            component = random.choice(list(boundary_components))
            if component not in selected_components and not any(graph.has_edge(node1, node2) for node1 in component for node2 in selected_nodes):
                selected_components.append(component)
                selected_nodes.update(component)
                boundary_components.remove(component)
                unchanged_counter = 0
            else:
                unchanged_counter += 1
                if unchanged_counter > 20:
                    break

        #move the selected components to a new district
        for component in selected_components:
            #if the component is the whole district, skip it as it wont produce a valid districting
            curr_district = dist_dict.get(next(iter(component)))
            if sum(graph.nodes[node]['voters'] for node in component) == dist_population_dict_proposal.get(curr_district):
                continue

            new_curr_district_nodes = {node for node in graph.nodes if dist_dict_proposal.get(node) == curr_district} - component
            #check if the district is still contiguous (do DFS)
            visited = set()
            stack = [next(iter(new_curr_district_nodes))]
            while stack:
                current = stack.pop()
                visited.add(current)
                stack.extend(neighbor for neighbor in graph.neighbors(current) if neighbor not in visited and neighbor in new_curr_district_nodes)

            if visited != new_curr_district_nodes:
                continue

            #get the district to move the component to
            connected_districts = {dist_dict.get(neighbor) for node in component for neighbor in graph.neighbors(node)} - {curr_district}
            if not connected_districts:
                continue

            new_district = random.choice(list(connected_districts))

            #move the component to the proposed district
            for node in component:
                dist_dict_proposal.update({node: new_district})
                dist_population_dict_proposal.update({new_district: dist_population_dict_proposal.get(new_district, 0) + graph.nodes[node]['voters']})
                dist_population_dict_proposal.update({dist_dict.get(node): dist_population_dict_proposal.get(curr_district, 0) - graph.nodes[node]['voters']})
            if steps > hot_steps:
                pop_eq_proposal = (pop_eq - abs(dist_population_dict.get(curr_district) - avg_voters) - abs(dist_population_dict.get(new_district) - avg_voters) + abs(dist_population_dict_proposal.get(curr_district) - avg_voters) + abs(dist_population_dict_proposal.get(new_district) - avg_voters))

        boundary_nodes_proposal = set()
        active_edges_proposal = set()
        for edge in dist_graph.edges:
            if dist_dict_proposal.get(edge[0]) != dist_dict_proposal.get(edge[1]):
                boundary_nodes_proposal.add(edge[0])
                boundary_nodes_proposal.add(edge[1])
                district_proposal.edges[edge]['is_active'] = False
            else:
                district_proposal.edges[edge]['is_active'] = True
                active_edges_proposal.add(edge)

        #calculate the weight of the proposal
        if steps > hot_steps:
            norm_pop_eq_proposal = pop_eq_proposal
            comp_proposal = (len(dist_graph.edges) - len(active_edges_proposal)) / len(dist_graph.edges)
            try:
                exponent = -pop_eq_beta * norm_pop_eq_proposal - comp_beta * comp_proposal
            except OverflowError:
                exponent = -80
            if prev_exponent == 0:
                    prev_exponent = exponent
            acceptance_probabilty = exp(min(max(-80, prev_exponent - exponent), 300))
            print(f"Difference: {prev_exponent - exponent}, Acceptance probability: {acceptance_probabilty}")

            if random.random() < acceptance_probabilty:
                print(f"Accepted at step {steps}/{annealing_steps + cold_steps + hot_steps}, p = {acceptance_probabilty}")
                dist_graph = deepcopy(district_proposal)
                dist_dict = dist_dict_proposal.copy()
                dist_population_dict = dist_population_dict_proposal.copy()
                boundary_nodes = boundary_nodes_proposal.copy()
                active_edges = active_edges_proposal.copy()
                pop_eq = pop_eq_proposal
                prev_exponent = exponent
                accepted += 1
            else:
                district_proposal = deepcopy(graph)
                dist_dict_proposal = dist_dict.copy()
                dist_population_dict_proposal = dist_population_dict.copy()
                print(f"Rejected at step {steps}/{annealing_steps + cold_steps + hot_steps}, p = {acceptance_probabilty}")
                rejected += 1
        #For the hot steps, accept the proposal every time
        else:
            print(f"Accepted at step {steps}/{annealing_steps + cold_steps + hot_steps}, p = 1")
            dist_graph = deepcopy(district_proposal)
            dist_dict = dist_dict_proposal.copy()
            dist_population_dict = dist_population_dict_proposal.copy()
            boundary_nodes = boundary_nodes_proposal.copy()
            active_edges = active_edges_proposal.copy()
            accepted += 1

        #update the parameters
        steps += 1
        if steps > hot_steps and steps <= hot_steps + annealing_steps:
            pop_eq_beta += pop_tol_target / annealing_steps
            comp_beta += comp_weight_target / annealing_steps
        
        if steps % 100 == 0:
            print(f"Step {steps}/{annealing_steps + cold_steps + hot_steps}")
            print(f"Accepted: {accepted}, Rejected: {rejected}")
            try:
                data[int(steps / 100), 0] = accepted
                data[int(steps / 100), 1] = rejected
            except IndexError:
                pass
            accepted = 0
            rejected = 0

    if len(set(dist_dict.values())) != number_of_districts:
        raise ValueError("Unexpected error in redistricting - number of districts is not correct")

    return [[node for node in dist_dict if dist_dict.get(node) == i] for i in range(number_of_districts)], districts, data

import math
import random
import time
from copy import deepcopy
import networkx as nx


def favouritism_alg(graph: nx.Graph,
                    number_of_districts: int,
                    parties_names: list,
                    party_to_favour: str,
                    deviation: float = 0.01,
                    convergence_acceptance: float = 0.05,
                    seed: int = int(time.time())) -> tuple:
    random.seed(seed)

    dist_graph = deepcopy(graph)

    # Get index of the party to favour
    party_index = parties_names.index(party_to_favour)

    # Seeding and Initialization
    best_wins = -1
    closest_votes_to_win = np.inf
    districts = []
    for i in range(500):  # Initial seeding attempts
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
            
    # Prepare district mappings and edge activity
    dist_dict = {}
    dist_population_dict = {}
    total_population = sum(graph.nodes[node].get('voters', 0) for node in graph.nodes)
    avg_voters = int(total_population / number_of_districts)

    for district_index, dist in enumerate(districts):
        for node in dist:
            dist_dict[node] = district_index
            dist_population_dict[district_index] = dist_population_dict.get(district_index, 0) + graph.nodes[node]['voters']
    
    best_dist_dict = dist_dict.copy()

    # Mark inactive edges
    not_active_edges = set()
    for edge in graph.edges:
        if dist_dict.get(edge[0]) != dist_dict.get(edge[1]):
            not_active_edges.add(edge)
            graph.edges[edge]['is_active'] = False
        else:
            graph.edges[edge]['is_active'] = True

    # Metrics Initialization
    voters_deviation = math.sqrt(sum((avg_voters - v)**2 for v in dist_population_dict.values()) / avg_voters)
    tau = 0.01  # Initial temperature
    step = 1
    accepted = 0
    accepted_sum = 0
    record_wins = best_wins
    data = []

    # Main Optimization Loop
    while True:
        if best_wins > record_wins:
            record_wins = best_wins
            best_dist_dict = dist_dict.copy()
            
        if step % 200 == 0:
            acceptance_rate = accepted / 200
            accepted_sum += acceptance_rate
            print(f"Acceptance rate: {acceptance_rate:.3f}, Step: {step}")
            print(f"Record Wins: {record_wins}, Best Wins: {best_wins}")
            data.append(acceptance_rate)
            accepted = 0

            # Convergence check
            if acceptance_rate <= convergence_acceptance and best_wins == record_wins:
                best_dist_dict = dist_dict.copy()
                break
            if step % 10000 == 0:
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

        delta_percentage_weight = 0.2  # Weight for the change in percentage support

        # Acceptance probability
        if delta_wins > 0:
            acceptance_probability = 1.0
        elif delta_percentage > delta_percentage_weight:
            acceptance_probability = math.exp(delta_percentage) * math.exp(delta_wins * 10)
        else:
            delta_deviation = new_voters_deviation - voters_deviation
            try:
                acceptance_probability = math.exp(-delta_deviation / tau) * math.exp(delta_wins * 1/(2*tau))
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
            best_wins += delta_wins
            accepted += 1

            # Update edge activity
            update_edge_activity(not_active_edges, graph, dist_dict)
        step += 1

    # Generate final districts
    final_districts = [[node for node in dist_dict if best_dist_dict[node] == i] for i in range(number_of_districts)]
    print (record_wins)
    return final_districts, data


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

    return math.sqrt(
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




import pickle as pkl
import geopandas as gpd

graph = pkl.load(open("pickle_files/graph_voters.pkl", "rb"))
# merged_gdf = gpd.read_file("temp/gminy_sejm_2023/gminy_sejm_2023.shp", encoding="utf-8")
# party_list = ["KOMIT_1", "KOMIT_2", "KOMIT_3", "KOMIT_4", "KOMIT_5", "KOMIT_6", "KOMIT_7", "KOMIT_8", "KOMIT_9", "KOMIT_10", "KOMIT_11", "KOMIT_12"]
# #add votes to the graph
# for index, row in merged_gdf.iterrows():
#     for party in party_list:
#         graph.nodes[row["JPT_KOD_JE"]][party] = row[party]

# favoured_party = "KOMIT_5" #Koalicja Obywatelska

# new,data = favouritism_alg(graph, 460, party_list, favoured_party, 0.01, 0.01)
# pkl.dump(new, open("temp/favouritism_alg.pkl", "wb"))
# pkl.dump(data, open("temp/favouritism_alg_data.pkl", "wb"))

# # new, old = graph_cut_algorithm(graph, 100, max_iterations=2000, T=200, num_of_chains=2, lambda_val=5, beta_end=3)

# threads = []
# results = [None] * 2

# def run_algorithm(index, *args):
#     results[index] = redist_flip_alg(*args)

# threads.append(threading.Thread(target=run_algorithm, args=(0, graph, 20, 10, 2000, 200, 0.05, 0.1, 100, 1)))
# #threads.append(threading.Thread(target=run_algorithm, args=(1, graph, 20, 100, 2000, 200, 0.05, 1, 0.1, 1)))

# for thread in threads:
#     thread.start()

# for thread in threads:
#     thread.join()

# new1, old1, data1 = results[0]
# # new2, old2, data2 = results[1]

# pkl.dump(new1, open("temp/graph_cut_algorithm_1.pkl", "wb"))
# pkl.dump(old1, open("temp/graph_cut_algorithm_start_1.pkl", "wb"))
# pkl.dump(data1, open("temp/redist_flip_alg_data_1.pkl", "wb"))

# pkl.dump(new2, open("temp/graph_cut_algorithm_2.pkl", "wb"))
# pkl.dump(old2, open("temp/graph_cut_algorithm_start_2.pkl", "wb"))
# pkl.dump(data2, open("temp/redist_flip_alg_data_2.pkl", "wb"))



new, old, data = redist_flip_alg(graph, 460, hot_steps=500, annealing_steps=2000, cold_steps=1000, lambda_prob=0.1, pop_tol_target=0.95, comp_weight_target=0.05)

pkl.dump(new, open("temp/graph_cut_algorithm.pkl", "wb"))
pkl.dump(old, open("temp/graph_cut_algorithm_start.pkl", "wb"))
pkl.dump(data, open("temp/redist_flip_alg_data.pkl", "wb"))