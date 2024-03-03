##
## Imports
##
from lib.graph import Graph
from lib.tsp.tspvisual import TSPVisual

##
## Execute the main function
##
if __name__ == "__main__":
    # Create a new graph
    graph = Graph.rand(num_nodes=7)

    # Visualize the graph
    TSPVisual.brute_force(graph)
