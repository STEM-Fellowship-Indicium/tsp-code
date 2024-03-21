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
    ## Check if it's a list of graphs or a map of graphs (with their ids as keys)
    ##
    if isinstance(graphs, list):
        ##
        ## Convert to a list of graphs (from the map) and return
        ##
        return [Graph.from_map(graph) for graph in graphs]

    elif isinstance(graphs, dict):
        ##
        ## Convert to a list of graphs (from the map) and return
        ##
        return [Graph.from_map(graph) for graph in graphs.values()]

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
    from lib.utils.export_graphs import export_graphs_list, export_graphs_map
    import os

    ##
    ## Generate graphs
    ##
    graphsN1 = generate_graphs(n=10, num_nodes=1)  ## 10 graphs

    ##
    ## Test A
    ##
    export_graphs_list(graphsN1, "data/tests/unit/import_graphs_list.json")
    graphs = import_graphs("data/tests/unit/import_graphs_list.json")

    for graph in graphs:
        print(graph)

    ##
    ## Test B
    ##
    export_graphs_map(graphs, "data/tests/unit/import_graphs_map.json")
    graphs = import_graphs("data/tests/unit/import_graphs_map.json")

    for graph in graphs:
        print(graph)

    ##
    ## Delete the file
    ##
    # os.remove("data/tests/unit/import_graphs_list.json")
    # os.remove("data/tests/unit/import_graphs_map.json")


##
## End of file
##
