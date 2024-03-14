##
## Imports
##
from lib.graph import Graph
from lib.tsp.tspvisual import TSPVisual
from lib.utils.generate_graphs import generate_graphs

##
## Execute the main function
##
if __name__ == "__main__":
    # Create a new graph
    # graph = Graph.rand(num_nodes=20)

    # Visualize the graph
    # TSPVisual.brute_force(graph)

    # Generate graphs
    graphs = generate_graphs()
    for graph in graphs:
        print(graph.shortest_tour)

##
## End of file
##
