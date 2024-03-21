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

    ## Create a new graph
    graph = Graph.rand(num_nodes=50)

    ## Visualize the graph
    shortest_tour = TSPAlgorithms.greedy_heuristic(graph)
    TSPVisual.two_opt(graph, shortest_tour)
##
## End of file
##
