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
from typing import List
import json


##
## Import graphs
##
def import_graphs(filename: str) -> List[Graph]:
    ##
    ## Open the file
    ##
    with open(filename, "r") as file:
        graphs = json.load(file)

    ##
    ## Create the graphs
    ##
    graphs = [Graph.from_map(graph) for graph in graphs]

    ##
    ## Return the graphs
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
    from lib.utils.generate_graphs import generate_graphs
    from lib.utils.export_graphs import export_graphs
    import os

    ##
    ## Generate graphs
    ##
    graphsN1 = generate_graphs(n=10, num_nodes=1)  ## 10 graphs

    ##
    ## Save to a file
    ##
    export_graphs(graphsN1, "data/tests/unit/import_graphs.json")

    ##
    ## Import the graphs
    ##
    graphs = import_graphs("data/tests/unit/import_graphs.json")

    ##
    ## Delete the file
    ##
    # os.remove("data/tests/unit/import_graphs.json")

    ##
    ## Print the graphs
    ##
    for graph in graphs:
        print(graph)

##
## End of file
##
