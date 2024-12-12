import networkx as nx
import numpy as np
import random
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

    number_of_districts : int
        The number of districts to create.

    deriviation : float, optional
        The deviation from the average number of voters in a district. 
        For 5% deviation, use 0.05. Default is 0.01 (1%).
        Derivation cannot be higher than 1.

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

    if deriviation > 1:
        raise ValueError("Derivation cannot be higher than 1.")
    
    graph_copy = graph.copy()

    #Calculate the average number of voters in a district
    total_voters = 0
    for node in graph.nodes:
        total_voters += graph.nodes[node]['voters']
    
    average_voters = int(total_voters / number_of_districts)

    #Every county with more than the average number of voters (+/- deriviation) is considered a district and is removed from the graph
    districts = []
    for node in graph.nodes:
        if graph.nodes[node]['voters'] > average_voters * (1 + abs(deriviation)):
            districts.append([node])
    for district in districts:
        graph.remove_node(district[0])

    number_of_districts -= len(districts)

    random.seed(seed)
    #Creating the remaining districts as long as there are nodes in the graph
    while len(graph.nodes) > number_of_districts and len(graph.nodes) > 0 and number_of_districts > 0:
        district = []
        #Start district from a random county
        district.append(random.choice(list(graph.nodes)))
        current_voters = graph.nodes[district[0]]['voters']

        #Add neighbors to the district until the number of voters is close to the average
        while current_voters < average_voters * (1 - deriviation) and len(graph.nodes) > number_of_districts:
            #Add all neighbors of the district
            neighbors = set()
            for node in district:
                neighbors.update(list(graph.neighbors(node)))
            
            #Remove nodes that are already in the district
            neighbors = neighbors - set(district)

            #District is as good as it gets
            if len(neighbors) == 0:
                break

            #Remove nodes that have too many voters
            neighbors_copy = neighbors.copy()
            for neighbor in neighbors_copy:
                if graph.nodes[neighbor]['voters'] + current_voters > average_voters * (1 + deriviation):
                    neighbors.remove(neighbor)

            #If there are no neighbors left, add the one that is the closest to the average number of voters
            #if the average number of voters is closer to the current number of voters with the best neighbor
            if len(neighbors) == 0:
                best_neighbor = None
                best_diff = abs(average_voters - current_voters)
                for neighbor in neighbors_copy:
                    diff = abs(graph.nodes[neighbor]['voters'] + current_voters - average_voters)
                    if diff < best_diff:
                        best_diff = diff
                        best_neighbor = neighbor
                
                if best_neighbor is not None:
                    district.append(best_neighbor)
                    current_voters += graph.nodes[best_neighbor]['voters']

                break
            
            #Try to find a neighbor the most compact to the rest of the district
            best_neighbor = None
            best_compactness = 0
            for neighbor in neighbors:
                compactness = 0
                for node in district:
                    compactness += 1 if neighbor in graph.neighbors(node) else 0
                if compactness > best_compactness:
                    best_compactness = compactness
                    best_neighbor = neighbor
            
            #Add the best neighbor to the district
            if best_neighbor is None:
                break

            district.append(best_neighbor)
            current_voters += graph.nodes[best_neighbor]['voters']
        
        #Remove the district from the graph
        for node in district:
            graph.remove_node(node)
        
        #Add the district to the list of districts
        districts.append(district)
        number_of_districts -= 1
        

    #Add the remaining nodes to the districts 
    if len(graph.nodes) > 0:
        for node in graph.nodes:
            #select a random neighbour of node
            neighbor = random.choice(list(graph_copy.neighbors(node)))

            #add node to the district of the neighbor
            for district in districts:
                if neighbor in district:
                    district.append(node)
                    break

    #If there are districts with only one node, add them to the list of districts
    if number_of_districts > 0:
        for node in graph.nodes:
            districts.append([node])
 
    return districts