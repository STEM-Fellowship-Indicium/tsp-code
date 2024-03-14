##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##
from lib.graph import Graph
from lib.tsp.tspalgorithms import TSPAlgorithms
from lib.types.tspalgorithm import TSPAlgorithm
from typing import List


##
## Generate graphs
##
def generate_graphs() -> List[Graph]:
    ##
    ## Generate a list of graphs
    ##
    ## Need atleast 1000 samples, also get best node route for each
    ##
    n = 1000
    num_nodes = 2
    graphs = [Graph.rand(num_nodes=num_nodes) for _ in range(n)]
    ##
    ## Set the shortest tour for each graph
    ##
    for graph in graphs:
        TSPAlgorithms.set_shortest_tour(graph, algorithm=TSPAlgorithm.BruteForce)

    ##
    ## Return the list of graphs
    ##
    return graphs

    ##
    ## End of function
    ##


##
## Test the function
##
if __name__ == "__main__":
    ##
    ## Generate graphs
    ##
    graphs = generate_graphs()

    ##
    ## Print the shortest tour for each graph
    ##
    for graph in graphs:
        print(graph.shortest_tour)
