import networkx as nx

def seeding_algorithm(graph: nx.Graph, number_of_districts: int, deriviation: float = 0.01) -> list:
    """
    Seeding algorithm for districting (source: https://doi.org/10.1016/S0962-6298(99)00047-5)

    @type graph: networkx.Graph
    @param graph: A graph representing the counties and their neighbors
    @type number_of_districts: int
    @param number_of_districts: The number of districts to create
    @type deriviation: float
    @param deriviation: The deriviation from the average number of voters in a district. For 5% deriviation, use 0.05 etc. Default = 1%
    @rtype: list
    @returns: List of districts. Each district is a list of counties' IDs (for district with one county, the list will contain one element)
    """

    #Calculate the average number of voters in a district
    total_voters = 0
    for node in graph.nodes:
        total_voters += graph.nodes[node]['voters']
    
    average_voters = int(total_voters / number_of_districts)
