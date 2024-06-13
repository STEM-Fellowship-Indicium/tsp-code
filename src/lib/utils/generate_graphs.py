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
from lib.interfaces.tspalgorithm import TSPAlgorithm
from typing import List
import json


##
## Generate graphs
##
def generate_graphs(
    n: int, num_nodes: int, algorithm: TSPAlgorithm = TSPAlgorithm.NoneType
) -> List[Graph]:
    ##
    ## Generate a list of graphs
    ##
    graphs = [Graph.rand(num_nodes=num_nodes) for _ in range(n)]

    ##
    ## Set the shortest tour for each graph
    ##
    if algorithm != TSPAlgorithm.NoneType:
        for graph in graphs:
            TSPAlgorithms.set_shortest_tour(graph, algorithm)

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
    ## Test imports
    ##
    from lib.utils.export_graphs import export_graphs
    import os

    ##
    ## Generate graphs
    ##
    graphsN3 = generate_graphs(
        n=100, num_nodes=3, algorithm=TSPAlgorithm.BruteForce
    )  ## 100 graphs

    ##
    ## Generate graphs
    ##
    graphsN5 = generate_graphs(
        n=100, num_nodes=5, algorithm=TSPAlgorithm.BruteForce
    )  ## 100 graphs

    ##
    ## Generate graphs
    ##
    graphsN10 = generate_graphs(
        n=100, num_nodes=7, algorithm=TSPAlgorithm.BruteForce
    )  ## 100 graphs

    ##
    ## Save to a file
    ##
    export_graphs(
        graphsN3 + graphsN5 + graphsN10, "data/tests/unit/generate_graphs.json"
    )

    ##
    ## Delete the file
    ##
    # os.remove("data/tests/unit/generate_graphs.json")


##
## End of file
##
