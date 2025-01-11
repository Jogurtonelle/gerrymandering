from copy import deepcopy
from math import exp
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
        for node in missing_nodes:
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
                    raise ValueError("Unexpected error in seeding algorithm - please try again with a different seed")


        #If there are districts with only one node, add them to the list of districts
        if number_of_districts > 0:
            for node in graph_copy.nodes:
                districts.append([node])

        return districts
 
    i = 100
    while i > 0:
        try:
            return seeding_algorithm_inner(graph, number_of_districts, deriviation, seed)
        except ValueError as e:
            i -= 1
            #print(f"Error in seeding algorithm - retrying {100 - i}/100")
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

def markov_simulated_annealing( graph: nx.Graph,
                                number_of_districts: int,
                                beta_1: float = 1,
                                beta_2: float = 1,
                                seed: int = int(time.time()),
                                ) -> list:
    
    districts = seeding_algorithm(graph, number_of_districts, seed=seed)
    dist_dict = {}
    dist_population_dict = {}
    dist_graph = nx.Graph(graph)
    try:
        avg_voters = sum(graph.nodes[node]['voters'] for node in graph.nodes) / number_of_districts
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
    
    not_active_edges = set()
    for edge in dist_graph.edges:
        if 'is_active' not in dist_graph.edges[edge]:
            dist_graph.edges[edge]['is_active'] = False
            not_active_edges.add(edge)
    
    #Getting nodes v such that N(v) & S_δ(S_delta) is not empty
    boundary_nodes = set()
    for edge in not_active_edges:
        boundary_nodes.add(edge[0])
        boundary_nodes.add(edge[1])

    comp = len(not_active_edges) / len(dist_graph.edges)
    weight = exp(-beta_1 * pop_eq - beta_2 * comp)

    district_proposal = deepcopy(dist_graph)
    dist_dict_proposal = dist_dict.copy()
    boundary_nodes_proposal = boundary_nodes.copy()
    not_active_edges_proposal = not_active_edges.copy()
    dist_population_dict_proposal = dist_population_dict.copy()
    while(True):
        #Randomly select a node v from the boundary nodes and an edge e from intersect of neighbors of v and not active edges
        v = random.choice(list(boundary_nodes))
        e = random.choice(list(set(dist_graph.neighbors(v)) & not_active_edges))

        #If the v's district is containing only one node, then skip the iteration (as it is not possible to move the node to create valid districts)
        if dist_population_dict.get(dist_dict.get(v)) == graph.nodes[v]['voters']:
            continue

        old_district = dist_dict.get(v)
        new_district = dist_dict.get(e[0]) if dist_dict.get(e[0]) != old_district else dist_dict.get(e[1])

        #Calculate the new population equality score
        pop_eq_proposal = pop_eq - abs(dist_population_dict.get(old_district) - avg_voters) - abs(dist_population_dict.get(new_district) - avg_voters) + abs(dist_population_dict.get(old_district) - graph.nodes[v]['voters'] - avg_voters) + abs(dist_population_dict.get(new_district) + graph.nodes[v]['voters'] - avg_voters)

        #Update the proposal
        dist_dict_proposal.update({v: new_district})
        dist_population_dict_proposal.update({old_district: dist_population_dict.get(old_district) - graph.nodes[v]['voters']})
        dist_population_dict_proposal.update({new_district: dist_population_dict.get(new_district) + graph.nodes[v]['voters']})

        for neighbor in dist_graph.neighbors(v):
            if dist_dict.get(neighbor) == old_district:
                boundary_nodes_proposal.add(neighbor)
                not_active_edges_proposal.update({(v, neighbor)})
                district_proposal.edges[v, neighbor]['is_active'] = False

            if dist_dict.get(neighbor) == new_district:
                district_proposal.edges[v, neighbor]['is_active'] = True
                not_active_edges_proposal.discard((v, neighbor))
                not_active_edges_proposal.discard((neighbor, v)) #two options as it can be in either form

                #Check if the neighbor is still a boundary node
                if all(dist_dict.get(neighbor) == new_district for neighbor in dist_graph.neighbors(neighbor)):
                    boundary_nodes_proposal.discard(neighbor)

        comp_proposal = len(not_active_edges_proposal) / len(dist_graph.edges)
        weight_proposal = exp(-beta_1 * pop_eq_proposal - beta_2 * comp_proposal)

        if random.random() < min(1, weight_proposal / weight):
            dist_graph = deepcopy(district_proposal)
            dist_dict = dist_dict_proposal.copy()
            dist_population_dict = dist_population_dict_proposal.copy()
            boundary_nodes = boundary_nodes_proposal.copy()
            not_active_edges = not_active_edges_proposal.copy()
            pop_eq = pop_eq_proposal
            weight = weight_proposal
        else:
            district_proposal = deepcopy(dist_graph)
            dist_dict_proposal = dist_dict.copy()
            boundary_nodes_proposal = boundary_nodes.copy()
            not_active_edges_proposal = not_active_edges.copy()
            dist_population_dict_proposal = dist_population_dict.copy()


def redist_flip_alg(graph: nx.Graph,
                    number_of_districts: int,
                    lambda_prob: float = 0.05,
                    pop_tol_target: float = 0.95,
                    pop_tol_increase: float = 0.005,
                    pop_tol_steps_before_increase: int = 10,
                    comp_weight_target: float = 0.5,
                    comp_weight_increase: float = 0.01,
                    cold_steps: int = 100,
                    seed: int = int(time.time()),
                    ) -> list:
    
    if lambda_prob < 0 or lambda_prob > 1:
        raise ValueError("Lambda probability has to be between 0 and 1")
    if number_of_districts < 1:
        raise ValueError("Number of districts has to be greater than 0")
    if pop_tol_target < 0 or pop_tol_target > 1:
        raise ValueError("Population tolerance target has to be between 0 and 1")
    if pop_tol_increase < 0 or pop_tol_increase > 1:
        raise ValueError("Population tolerance decrease has to be between 0 and 1")
    if pop_tol_steps_before_increase < 1:
        raise ValueError("Population tolerance steps before decrease has to be greater than 0")
    if comp_weight_target < 0 or comp_weight_target > 1:
        raise ValueError("Compactness weight target has to be between 0 and 1")
    if comp_weight_increase < 0 or comp_weight_increase > 1:
        raise ValueError("Compactness weight increase has to be between 0 and 1")
    
    beta_1, beta_2 = 0, 0 #TODO
    
    districts = seeding_algorithm(graph, number_of_districts, seed=seed)
    dist_dict = {}
    dist_population_dict = {}
    dist_graph = nx.Graph(graph)
    try:
        avg_voters = sum(graph.nodes[node]['voters'] for node in graph.nodes) / number_of_districts
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
    
    not_active_edges = set()
    for edge in dist_graph.edges:
        if 'is_active' not in dist_graph.edges[edge]:
            dist_graph.edges[edge]['is_active'] = False
            not_active_edges.add(edge)

    active_edges = set(graph.edges) - not_active_edges
    
    #Getting boundary nodes - v such that N(v) & S_δ(S_delta) is not empty
    boundary_nodes = set()
    for edge in not_active_edges:
        boundary_nodes.add(edge[0])
        boundary_nodes.add(edge[1])

    comp = len(not_active_edges) / len(dist_graph.edges)
    weight = exp(-beta_1 * pop_eq - beta_2 * comp)

    district_proposal = deepcopy(dist_graph)
    dist_dict_proposal = dist_dict.copy()
    boundary_nodes_proposal = boundary_nodes.copy()
    not_active_edges_proposal = not_active_edges.copy()
    active_edges_proposal = active_edges.copy()
    dist_population_dict_proposal = dist_population_dict.copy()
    pop_eq_proposal = pop_eq


    pop_tol_steps = pop_tol_target / pop_tol_increase * pop_tol_steps_before_increase
    comp_weight_steps_before_increase = int(pop_tol_steps / (comp_weight_target / comp_weight_increase))
    pop_eq_beta = 0
    comp_beta = 0
    steps = 0
    print(f"Starting weight: {weight}")
    while(steps < pop_tol_steps + cold_steps):
        timer = time.time()
        #set flipped edges
        flipped_edges = set(edge for edge in active_edges if random.random() <= lambda_prob)
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(graph.nodes)

        #TODO
        if len(temp_graph.edges) != 0:
            raise ValueError("Graph has to have edges - debug")
        
        for edge in flipped_edges:
            temp_graph.add_edge(edge[0], edge[1])

        #get boundary connected components
        boundary_components = []
        for component in nx.connected_components(temp_graph):
            if any(node in boundary_nodes for node in component):
                boundary_components.append(component)

        #select a subset of boundary components not connecting to each other
        R = random.randint(0, len(boundary_components))
        selected_components = []
        selected_nodes = set()
        unchanged_counter = 0
        while len(selected_components) < R:
            component = random.choice(list(boundary_components))
            if component not in selected_components and not any(graph.has_edge(node1, node2) for node1 in component for node2 in selected_nodes):
                selected_components.append(component)
                selected_nodes.update(component)
                boundary_components.remove(component)
                unchanged_counter = 0
            else:
                unchanged_counter += 1
                if unchanged_counter > 50:
                    break

        #move the selected components to a new district
        for component in selected_components:
            #if the component is the whole district, skip it as it wont produce a valid districting
            if sum(graph.nodes[node]['voters'] for node in component) == dist_population_dict.get(dist_dict.get(next(iter(component)))):
                continue

            #get the district to move the component to
            connected_districts = set(dist_dict.get(neighbor) for node in component for neighbor in graph.neighbors(node))
            if len(connected_districts) == 0:
                continue
            new_district = random.choice(list(connected_districts))
            curr_district = dist_dict.get(next(iter(component)))

            #move the component to the proposed district
            for node in component:
                dist_dict_proposal.update({node: new_district})
                dist_population_dict_proposal.update({new_district: dist_population_dict_proposal.get(new_district, 0) + graph.nodes[node]['voters']})
                dist_population_dict_proposal.update({dist_dict.get(node): dist_population_dict_proposal.get(curr_district, 0) - graph.nodes[node]['voters']})
            pop_eq_proposal = pop_eq - abs(dist_population_dict.get(curr_district) - avg_voters) - abs(dist_population_dict.get(new_district) - avg_voters) + abs(dist_population_dict_proposal.get(curr_district) - avg_voters) + abs(dist_population_dict_proposal.get(new_district) - avg_voters)

            #update the connections in the proposed district and the current district
            new_district_nodes = set(node for node in dist_dict_proposal if dist_dict_proposal.get(node) == new_district)
            olf_district_nodes = set(node for node in dist_dict_proposal if dist_dict_proposal.get(node) == curr_district)

            for node in new_district_nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor in new_district_nodes:
                        district_proposal.edges[node, neighbor]['is_active'] = True
                        if not (node, neighbor) in active_edges_proposal and not (neighbor, node) in active_edges_proposal:
                            active_edges_proposal.add((node, neighbor))
                        if (node, neighbor) in not_active_edges_proposal or (neighbor, node) in not_active_edges_proposal:
                            not_active_edges_proposal.discard((node, neighbor))
                            not_active_edges_proposal.discard((neighbor, node))
                    else:
                        district_proposal.edges[node, neighbor]['is_active'] = False
                        if (node, neighbor) in active_edges_proposal or (neighbor, node) in active_edges_proposal:
                            active_edges_proposal.discard((node, neighbor))
                            active_edges_proposal.discard((neighbor, node))
                        if not (node, neighbor) in not_active_edges_proposal and not (neighbor, node) in not_active_edges_proposal:
                            not_active_edges_proposal.add((node, neighbor))

                if any(neighbor not in new_district_nodes for neighbor in graph.neighbors(node)):
                    boundary_nodes_proposal.add(node)
            
            for node in olf_district_nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor in olf_district_nodes:
                        district_proposal.edges[node, neighbor]['is_active'] = True
                        if not (node, neighbor) in active_edges_proposal and not (neighbor, node) in active_edges_proposal:
                            active_edges_proposal.add((node, neighbor))
                        if (node, neighbor) in not_active_edges_proposal or (neighbor, node) in not_active_edges_proposal:
                            not_active_edges_proposal.discard((node, neighbor))
                            not_active_edges_proposal.discard((neighbor, node))
                    else:
                        district_proposal.edges[node, neighbor]['is_active'] = False
                        if (node, neighbor) in active_edges_proposal or (neighbor, node) in active_edges_proposal:
                            active_edges_proposal.discard((node, neighbor))
                            active_edges_proposal.discard((neighbor, node))
                        if not (node, neighbor) in not_active_edges_proposal and not (neighbor, node) in not_active_edges_proposal:
                            not_active_edges_proposal.add((node, neighbor))

                if any(neighbor not in olf_district_nodes for neighbor in graph.neighbors(node)):
                    boundary_nodes_proposal.add(node)

        #calculate the weight of the proposal
        comp_proposal = len(not_active_edges_proposal) / len(dist_graph.edges)
        weight_proposal = exp(max(-80, -pop_eq_beta * pop_eq_proposal - comp_beta * comp_proposal))

        if random.random() < min(1, weight_proposal / weight):
            dist_graph = deepcopy(district_proposal)
            dist_dict = dist_dict_proposal.copy()
            dist_population_dict = dist_population_dict_proposal.copy()
            boundary_nodes = boundary_nodes_proposal.copy()
            not_active_edges = not_active_edges_proposal.copy()
            pop_eq = pop_eq_proposal
            weight = weight_proposal
            print(f"Accepted at step {steps}/{pop_tol_steps + cold_steps}")
            print(weight_proposal/weight)
        else:
            district_proposal = deepcopy(dist_graph)
            dist_dict_proposal = dist_dict.copy()
            boundary_nodes_proposal = boundary_nodes.copy()
            not_active_edges_proposal = not_active_edges.copy()
            dist_population_dict_proposal = dist_population_dict.copy()
            print(f"Rejected at step {steps}/{pop_tol_steps + cold_steps}")
            print(weight_proposal/weight)
            print(-pop_eq_beta * pop_eq_proposal - comp_beta * comp_proposal)
            print("pop_eq", pop_eq, pop_eq_proposal)

        #update the parameters
        steps += 1
        if steps < pop_tol_steps:
            if steps % pop_tol_steps_before_increase == 0:
                pop_eq_beta += pop_tol_increase
                print(pop_eq_beta, "pop")
            if steps % comp_weight_steps_before_increase == 0:
                comp_beta += comp_weight_increase
                print(comp_beta, "comp")

    if len(dist_dict.values()) != number_of_districts:
        raise ValueError("Unexpected error in redistricting - number of districts is not correct")

    return [[node for node in dist_dict if dist_dict.get(node) == i] for i in range(number_of_districts)], districts

        

import pickle as pkl
graph = pkl.load(open("pickle_files/graph_voters.pkl", "rb"))
# new, old = graph_cut_algorithm(graph, 100, max_iterations=2000, T=200, num_of_chains=2, lambda_val=5, beta_end=3)

new, old = redist_flip_alg(graph, 100)

pkl.dump(new, open("temp/graph_cut_algorithm.pkl", "wb"))
pkl.dump(old, open("temp/graph_cut_algorithm_start.pkl", "wb"))