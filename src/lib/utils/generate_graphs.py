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
import json


##
## Generate graphs
##
def generate_graphs(n: int, num_nodes: int) -> List[Graph]:
    ##
    ## Generate a list of graphs
    ##
    ## Need atleast 1000 samples, also get best node route for each
    ##
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
    graphsN3 = generate_graphs(n=100, num_nodes=3)  ## 100 graphs

    ##
    ## Generate graphs
    ##
    graphsN5 = generate_graphs(n=100, num_nodes=5)  ## 50 graphs

    ##
    ## Generate graphs
    ##
    graphsN10 = generate_graphs(n=100, num_nodes=7)  ## 25 graphs

    ##
    ## Save to a file
    ##
    graphs_as_maps = [graph.to_map() for graph in graphsN3 + graphsN5 + graphsN10]
    with open("data/graphs.json", "w") as file:
        json.dump(graphs_as_maps, file, indent=4)
