import copy
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

    #random.seed(seed)
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

    seed : int, optional
        Random seed. Default is the current time.

    Returns
    -------
    list
        List of districts, where each district is a list of node IDs.
    """
    random.seed(seed)

    # Initial districts from the seeding algorithm (pi_0)
    starting_districts = seeding_algorithm(graph, number_of_districts, deriviation, seed)

    # Convert initial districts into subgraph objects
    subgraphs = []
    for district in starting_districts:
        subgraph = nx.Graph()
        for node in district:
            subgraph.add_node(node, voters=graph.nodes[node]['voters'])
            for neighbor in graph.neighbors(node):
                if neighbor in district:
                    subgraph.add_edge(node, neighbor)
        subgraphs.append(subgraph)

    # Helper function to compute total voters in a district
    def compute_total_voters(subgraph):
        return sum(subgraph.nodes[node]['voters'] for node in subgraph.nodes)

    # Initialize parameters for the algorithm
    q = 0.05  # Probability to "turn on" edges
    max_iterations = 1000  # Maximum number of MCMC iterations
    current_districts = starting_districts.copy()

    for iteration in range(max_iterations):
        # Step 1: "Turn on" edges
        edge_set = set()
        threads = []
        
        # Helper function to turn on edges in one subgraph (parallelizable)
        def turn_on_edges(subgraph, edge_set):
            for node in subgraph.nodes:
                for neighbor in graph.neighbors(node):
                    if neighbor in subgraph.nodes and random.random() < q:
                        edge_set.add((node, neighbor))

        # Turn on edges in parallel
        for subgraph in subgraphs:
            thread = threading.Thread(target=turn_on_edges, args=(subgraph, edge_set))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
            

        # Step 2: Identify boundary connected components
        boundary_nodes = set()
        threads = []

        # Helper function to identify boundary nodes in one subgraph (parallelizable)
        def identify_boundary_nodes(subgraph, boundary_nodes):
            for node in subgraph.nodes:
                if any(neighbor not in subgraph.nodes for neighbor in graph.neighbors(node)):
                    boundary_nodes.add(node)

        # Identify boundary nodes in parallel
        for subgraph in subgraphs:
            thread = threading.Thread(target=identify_boundary_nodes, args=(subgraph, boundary_nodes))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        boundary_components = []
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
                            if neighbor in boundary_nodes and neighbor not in visited
                        )
                boundary_components.append(component)

        # Step 3: Select nonadjacent connected components
        R = random.randint(1, len(boundary_components))
        selected_components = []
        for i in range(R):
            component = random.choice(boundary_components)
            if all(
                not any(neighbor in subgraph.nodes for neighbor in graph.neighbors(node))
                for node in component
                for subgraph in subgraphs
            ):
                selected_components.append(component)
            else:
                i -= 1

        # Step 4: Propose district swaps
        proposed_districts = current_districts.copy()
        for component in selected_components:
            candidate_districts = [d for d in current_districts if not set(component).intersection(d)]
            if candidate_districts:
                target_district = random.choice(candidate_districts)
                for node in component:
                    for district in proposed_districts:
                        if node in district:
                            district.remove(node)
                            break
                    target_district.append(node)

        # Step 5: Accept or reject the proposal
        current_population_variance = sum(
            abs(compute_total_voters(subgraph) - sum(graph.nodes[node]['voters'] for node in graph) / number_of_districts)
            for subgraph in subgraphs
        )
        proposed_population_variance = sum(
            abs(compute_total_voters(nx.subgraph(graph, district)) - sum(graph.nodes[node]['voters'] for node in graph) / number_of_districts)
            for district in proposed_districts
        )

        acceptance_ratio = min(1, proposed_population_variance / current_population_variance)
        if random.random() < acceptance_ratio:
            current_districts = proposed_districts

    return current_districts




import pickle as pkl
graph = pkl.load(open("pickle_files/graph_voters.pkl", "rb"))
for i in range(100):
    dist = graph_cut_algorithm(graph, 460)
    sum_of_nodes = 0
    for d in dist:
        sum_of_nodes += len(d.nodes)

    print(i, " ", sum_of_nodes) if sum_of_nodes != 2477 else print(i," OK")
