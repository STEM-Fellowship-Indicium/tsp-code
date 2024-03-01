##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


##
## Imports
##
import json
from lib.node import Node


##
## Edge class
##
## This class represents an edge in a graph.
##
class Edge:
    ##
    ## Constructor
    ##
    def __init__(self, index: int, start: Node, end: Node) -> None:
        """Initializer for the Edge class

        Args:
            index (int): The index of the edge
            start (Node): The start node of the edge
            end (Node): The end node of the edge
        """
        self.index = index
        self.start = start
        self.end = end

        ##
        ## End of function
        ##

    ##
    ## String representation of the edge
    ##
    def __str__(self) -> str:
        """String representation of the edge

        Returns:
            _type_: The string representation of the edge
        """
        return f"{self.start} -> {self.end}"

        ##
        ## End of function
        ##

    ##
    ## Convert the edge to a map
    ##
    def to_map(self) -> dict:
        """Convert the edge to a map

        Returns:
            dict: The map of the edge
        """
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
        """Convert the edge to a json map

        Returns:
            str: The json map of the edge
        """
        return json.dumps(self.to_map())

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## Execute the test
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
