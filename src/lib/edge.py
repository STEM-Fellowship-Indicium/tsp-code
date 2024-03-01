##
## Imports
##
import json
from node import Node


##
## Edge class
##
## This class represents an edge in a graph.
##
class Edge:
    ##
    ## Constructor
    ##
    def __init__(self, index: int, start: Node, end: Node):
        self.index = index
        self.start = start
        self.end = end

        ##
        ## End of function
        ##

    ##
    ## String representation of the edge
    ##
    def __str__(self):
        return f"{self.start} -> {self.end}"

        ##
        ## End of function
        ##

    ##
    ## Convert the edge to a map
    ##
    def to_map(self) -> dict:
        return {
            "index": self.index,
            "start": self.start.to_map(),
            "end": self.end.to_map(),
        }

        ##
        ## End of function
        ##

    ##
    ## Convert the edge to a json map
    ##
    def to_json(self) -> str:
        return json.dumps(self.to_map())

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the edge class only if we're executing THIS current file.
##
## This is so that if we import the Edge class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    n1 = Node(1, 0, 0)
    n2 = Node(2, 1, 1)
    e = Edge(1, n1, n2)
    print(e.to_json())
    print(str(e))
    print(e.start)
    print(e.end)

##
## End of file
##
