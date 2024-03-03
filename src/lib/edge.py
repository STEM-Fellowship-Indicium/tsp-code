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
    def __init__(self, idx: int, start: Node, end: Node, weight: int = 1) -> None:
        """Initializer for the Edge class

        Args:
            idx (int): The idx of the edge
            start (Node): The start node of the edge
            end (Node): The end node of the edge
        """
        self.idx = idx
        self.start = start
        self.end = end
        self.weight = weight

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
        return f"{self.start} --{self.weight}--> {self.end}"

        ##
        ## End of function
        ##

    ##
    ## Edge from map
    ##
    @staticmethod
    def from_map(map: dict = None) -> "Edge":
        """Create an edge from a map

        Args:
            map (dict): The map to create the edge from

        Returns:
            Edge: The edge created from the map
        """
        if map is None:
            return None

        return Edge(
            map["idx"],
            Node.from_map(map["start"]),
            Node.from_map(map["end"]),
            map["weight"],
        )

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
            "idx": self.idx,
            "weight": self.weight,
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
    ## Normalize the edge
    ##
    def normalize(self) -> None:
        """Normalize the edge"""
        self.start.normalize()
        self.end.normalize()

        ##
        ## End of function
        ##

    ##
    ## Print the edge
    ##
    def print(self) -> None:
        """Print the edge"""
        print(f"Edge {self.idx}: {self.start} --{self.weight}--> {self.end}")

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
    e = Edge(1, n1, n2, weight=10)
    print(e.to_json())
    print(str(e))
    print(e.start)
    print(e.end)

    e.print()

##
## End of file
##
