##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##
from typing import List
from lib.node import Node
from lib.types.tspalgorithm import TSPAlgorithm
import json


##
## Tour class
##
class Tour:
    ##
    ## Constructor
    ##
    def __init__(
        self,
        nodes: List[Node] = [],
        distance: int = 0,
        algorithm: str = TSPAlgorithm.NoneType,
    ) -> None:
        """Initializer for the Tour class

        Args:
            nodes (List[Node], optional): The nodes of the tour. Defaults to [].
            algorithm (TSPAlgorithm, optional): The algorithm used to solve the tour. Defaults to TSPAlgorithm.NoneType.
        """
        self.nodes = nodes
        self.algorithm = algorithm
        self.distance = distance

        ##
        ## End of function
        ##

    ##
    ## String representation of the tour
    ##
    def __str__(self) -> str:
        """String representation of the tour

        Returns:
            str: The string representation of the tour
        """
        return f"{[str(node) for node in self.nodes]}"

        ##
        ## End of function
        ##

    ##
    ## Import tour from map
    ##
    @staticmethod
    def from_map(map: dict = None) -> "Tour":
        """Import the tour from a map

        Args:
            map (dict): The map to import the tour from

        Returns:
            Tour: The tour from the json string
        """
        if map is None:
            return None

        nodes = [Node.from_map(node) for node in map["nodes"]]
        distance = map["distance"]
        algorithm = map["algorithm"]

        return Tour(nodes, distance, algorithm)

        ##
        ## End of function
        ##

    ##
    ## Convert the tour to a dictionary
    ##
    def to_map(self) -> dict:
        """Convert the tour to a dictionary

        Returns:
            dict: The dictionary representation of the tour
        """
        return {
            "nodes": [node.to_map() for node in self.nodes],
            "algorithm": self.algorithm,
            "distance": self.distance,
        }

        ##
        ## End of function
        ##

    ##
    ## Convert the tour to json
    ##
    def to_json(self) -> str:
        """Convert the tour to json

        Returns:
            str: The json representation of the tour
        """
        return json.dumps(self.to_map(), indent=4)

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## Test the Tour class
##
if __name__ == "__main__":
    nodes = [Node(0, 0, 0), Node(1, 1, 1), Node(2, 2, 2)]
    tour = Tour(nodes, 10, TSPAlgorithm.BruteForce)
    print(tour)

##
## End of file
##
