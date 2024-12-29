import copy
import networkx as nx
import numpy as np
import random
import time
import threading
import math
import time

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
    
    graph_copy = copy.deepcopy(graph)

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
 
    covered_nodes = {node for district in districts for node in district}
    missing_nodes = set(graph.nodes) - covered_nodes
    extra_nodes = covered_nodes - set(graph.nodes)

    if missing_nodes:
        print(f"Missing nodes: {missing_nodes}")
    if extra_nodes:
        print(f"Extra nodes: {extra_nodes}")

    return districts

def graph_cut_algorithm(graph: nx.Graph,
                        number_of_districts: int,
                        deriviation: float = 0.01,
                        lambda_val: int = 10,
                        beta_start: int = 1,
                        beta_end: int = 10,
                        q: float = 0.05,
                        num_of_chains: int = 5,
                        max_iterations: int = 200,
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

    def cumulative_distribution(lam, k, result_list):
        result = 0
        for i in range(k + 1):
            result += lam**i / math.factorial(i)
        result_list.append(np.exp(-lam) * result)

    def g_beta(districts, beta):
        result = 0
        for district in districts:
            result += (sum(graph.nodes[node]['voters'] for node in district)/average_voters) - 1
        return np.exp(-beta * result)

    random.seed(seed)

    # Initial districts from the seeding algorithm (pi_0)
    starting_districts = seeding_algorithm(graph, number_of_districts, deriviation, seed)
    average_voters = sum(graph.nodes[node]['voters'] for node in graph.nodes) / number_of_districts

    # Convert initial districts into subgraph objects
    current_districts = []
    for district in starting_districts:
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
        print(f"Starting chain with beta = {beta}", random.random())
        for iteration in range(iterations):
            threads = []

            # Step 1: "Turn on" edges
            edge_set = set()
            
            thread = threading.Thread(target=turn_on_edges, args=(edge_set, _current_districts, q))
            thread.start()
            threads.append(thread)
                
            # Step 2: Identify boundary connected components
            boundary_nodes = set()

            thread = threading.Thread(target=identify_boundary_nodes, args=(_current_districts, boundary_nodes))
            thread.start()
            threads.append(thread)

            for thread in threads:
                thread.join()

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
                R = int(np.random.poisson(lam=10))

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
            proposed_districts = copy.deepcopy(_current_districts)
            proposed_node_to_subgraph = _node_to_subgraph.copy()

            for component in selected_components:
                #get a random district to move the component to, which is connected to the component
                connected_districts = set()
                for node in component:
                    for neighbor in graph.neighbors(node):
                        if neighbor in _node_to_subgraph:
                            connected_districts.add(_node_to_subgraph.get(neighbor))
                
                #remove the district from the connected_districts set
                current_district = _node_to_subgraph.get(next(iter(component)))
                connected_districts.discard(current_district)

                #if there are no connected districts, skip the component (in typical case, this should not happen)
                if len(connected_districts) == 0:
                    continue

                #get the district to move the component to
                proposed_district = random.choice(list(connected_districts))

                #move the component to the proposed district
                for node in component:
                    proposed_districts[current_district].remove_node(node)
                    proposed_districts[proposed_district].add_node(node, voters=graph.nodes[node]['voters'])
                    proposed_node_to_subgraph.update({node: proposed_district})

                #update connections in the proposed district according to the graph
                for node in component:
                    for neighbor in graph.neighbors(node):
                        if neighbor in proposed_districts[proposed_district]:
                            proposed_districts[proposed_district].add_edge(node, neighbor)

            # Step 5: Accept or reject the proposal

            #get B_CP_pi_prime
            boundary_nodes_prime = set()

            identify_boundary_nodes(proposed_districts, boundary_nodes_prime)
            
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

            F_B_CP_pi = 0
            F_B_CP_pi_prime = 0
            
            #get F(∣B(CP,π)∣) where F - cumulative distribution
            F_B_CP_pi_list = []
            thread_pi = threading.Thread(target=cumulative_distribution, args=(lambda_val, len(B_CP_pi), F_B_CP_pi_list))
            thread_pi.start()

            #get F(∣B(CP,π′)∣) where F - cumulative distribution
            F_B_CP_pi_prime_list = []
            thread_pi_prime = threading.Thread(target=cumulative_distribution, args=(lambda_val, len(B_CP_pi_prime), F_B_CP_pi_prime_list))
            thread_pi_prime.start()

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
                for neighbor in proposed_districts[district].neighbors(node):
                    if neighbor not in nodes_vcp:
                        C_pi_prime_V_cp.add((node, neighbor))
            
            thread_pi.join()
            F_B_CP_pi = F_B_CP_pi_list[0]
            thread_pi_prime.join()
            F_B_CP_pi_prime = F_B_CP_pi_prime_list[0]

            #get the acceptance probability
            acceptance_probability = ((len(B_CP_pi)/len(B_CP_pi_prime))**R) * (F_B_CP_pi/F_B_CP_pi_prime) * (((1-q)**len(C_pi_prime_V_cp)) / ((1-q)**len(C_pi_V_cp))) * (g_beta(proposed_districts, beta) / g_beta(_current_districts, beta))
            acceptance_probability = min(1, acceptance_probability)

            if random.random() <= acceptance_probability:

                _current_districts.clear()
                _current_districts.extend(proposed_districts)

                _node_to_subgraph.clear()
                _node_to_subgraph.update(proposed_node_to_subgraph)

            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}")

    chains = []
    chain_current_districts = [copy.deepcopy(current_districts) for i in range(num_of_chains)]
    chain_node_to_subgraph = [node_to_subgraph.copy() for i in range(num_of_chains)]
    for i in range(num_of_chains):
        chains.append(threading.Thread(target=main_algorithm, args=(random.randint(beta_start, beta_end), max_iterations, chain_current_districts[i], chain_node_to_subgraph[i])))

    for chain in chains:
        chain.start()

    for chain in chains:
        chain.join()

    #get the final districts with the smallest variance of voters
    best_proposal_index = -1
    best_proposal_score = np.var([sum(graph.nodes[node]['voters'] for node in district.nodes) for district in current_districts])
    print(f"Starting score: {best_proposal_score}")
    for i, proposal in enumerate(chain_current_districts):
        score = np.var([sum(graph.nodes[node]['voters'] for node in district.nodes) for district in proposal])
        print(f"Chain {i} score: {score}")
        if score < best_proposal_score:
            best_proposal_score = score
            best_proposal_index = i

    if best_proposal_index == -1:
        return starting_districts, starting_districts
    
    districts = []
    for district in chain_current_districts[best_proposal_index]:
        districts.append(list(district.nodes))
    
    return districts, starting_districts


import pickle as pkl
graph = pkl.load(open("pickle_files/graph_voters.pkl", "rb"))
new, old = graph_cut_algorithm(graph, 460, max_iterations=200)

pkl.dump(new, open("temp/graph_cut_algorithm.pkl", "wb"))
pkl.dump(old, open("temp/graph_cut_algorithm_start.pkl", "wb"))
