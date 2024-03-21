##
## Imports
##
from lib.graph import Graph
from lib.tsp.tspvisual import TSPVisual
from lib.utils.generate_graphs import generate_graphs
from lib.types.tspalgorithm import TSPAlgorithm

##
## Execute the main function
##
if __name__ == "__main__":
    # Generate graphs
    graphs = generate_graphs(n=10, num_nodes=5, algorithm=TSPAlgorithm.two_opt)
    for graph in graphs:
        print(f"Graph Shortest Tour: {graph.shortest_tour}")


    # Create a new graph
    graph = Graph.rand(num_nodes=20)
    graph_two = graph.copy()

    # Visualize the graph
    TSPVisual.brute_force(graph)
    TSPVisual.two_opt(graph_two)
##
## End of file
##
