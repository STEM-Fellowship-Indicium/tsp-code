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
from typing import List, Dict, Union
import json


##
## Export graphs as either a list or map
##
def export_graphs(
    graphs: List[Graph], filename: str, type: Union[List, Dict] = List
) -> None:
    ##
    ## Save to a file
    ##
    with open(filename, "w") as file:
        ##
        ## Check the type
        ##
        if type == List:
            json.dump([graph.to_map() for graph in graphs], file, indent=4)

        elif type == Dict:
            json.dump({graph.id: graph.to_map() for graph in graphs}, file, indent=4)

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
    import os

    ##
    ## Generate graphs
    ##
    graphsN1 = generate_graphs(n=10, num_nodes=1)  ## 10 graphs

    ##
    ## Save to a file
    ##
    export_graphs(graphsN1, "data/tests/unit/export_graphs_list.json", type=List)
    export_graphs(graphsN1, "data/tests/unit/export_graphs_map.json", type=Dict)

    ##
    ## Delete the file
    ##
    # os.remove("data/tests/unit/export_graphs.json")

##
## End of file
##
