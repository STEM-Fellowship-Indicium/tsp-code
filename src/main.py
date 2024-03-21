##
## Imports
##
from lib.graph import Graph
from lib.tsp.tspvisual import TSPVisual
from lib.utils.generate_graphs import generate_graphs
from lib.types.tspalgorithm import TSPAlgorithm
from lib.tsp.tspalgorithms import TSPAlgorithms

##
## Execute the main function
##
if __name__ == "__main__":
    ## Generate graphs
    graphs = generate_graphs(n=10, num_nodes=5, algorithm=TSPAlgorithm.BruteForce)
    for graph in graphs:
        print(f"Graph Shortest Tour: {graph.shortest_tour}")

    ##
    ## Create a new graph
    ##
    ## If the graph already exists in a cache, we can also use:
    ##  Graph.from_cache(filename, id_of_graph_to_get)
    ##
    graph = Graph.rand(num_nodes=7)

    ##
    ## Visualize the graph and the shortest tour (from the greedy heuristic)
    ##
    print("Greedy heuristic visual")
    shortest_tour = TSPAlgorithms.greedy_heuristic(graph)
    graph.draw([shortest_tour], ["red"])

    ##
    ## Visualize the graph and the shortest tour (from the 2-opt algorithm)
    ##
    print("Two-opt visual")
    shortest_tour = TSPAlgorithms.two_opt(graph, shortest_tour)
    graph.draw([shortest_tour], ["green"])

    ##
    ## Visualize the graph and the shortest tour (from the brute force algorithm)
    ##
    print("Brute force visual")
    shortest_tour = TSPAlgorithms.brute_force(graph)
    graph.draw([shortest_tour], ["blue"])

    ##
    ## Visualizations
    ##
    ## TSPVisual.greedy_heuristic(graph)
    TSPVisual.two_opt(graph)
    ## TSPVisual.brute_force(graph)

##
## End of file
##
