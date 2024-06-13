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
from typing import List
from torch import Tensor
import numpy as np
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
    def __init__(self, idx: int, start: Node, end: Node, weight: float = 1.0) -> None:
        """Initializer for the Edge class

        Args:
            idx (int): The idx of the edge
            start (Node): The start node of the edge
            end (Node): The end node of the edge
            weight (float, optional): The weight of the edge. Defaults to 1.0.
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
    ## Convert the edge to a tensor
    ##
    def to_tensor(self) -> Tensor:
        """Convert the edge to a tensor

        Returns:
            Tensor: The tensor representation of the edge
        """
        start_np = self.start.to_numpy()
        end_np = self.end.to_numpy()

        return Tensor(np.array([start_np, end_np]))

        ##
        ## End of function
        ##

    ##
    ## Convert the edge to a numpy array
    ##
    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        """Convert the edge to a numpy array

        Returns:
            np.ndarray: The numpy array representation of the edge
        """
        start_np = self.start.to_numpy()
        end_np = self.end.to_numpy()

        return np.array([start_np, end_np], dtype=dtype)

        ##
        ## End of function
        ##

    ##
    ## Normalize the edge
    ##
    def normalize(
        self, min: List[float] = [0, 0], max: List[float] = [100, 100]
    ) -> None:
        """Normalize the edge"""
        self.start.normalize(min, max)
        self.end.normalize(min, max)

        ##
        ## End of function
        ##

    ##
    ## Denormalize the edge
    ##
    def denormalize(
        self, min: List[float] = [0, 0], max: List[float] = [100, 100]
    ) -> None:
        """Denormalize the edge"""
        self.start.denormalize(min, max)
        self.end.denormalize(min, max)

        ##
        ## End of function
        ##

    ##
    ## Create a copy of the edge
    ##
    def copy(self) -> "Edge":
        """Create a copy of the edge

        Returns:
            Edge: The copy of the edge
        """
        return Edge(self.idx, self.start.copy(), self.end.copy(), self.weight)

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

    print(e.to_tensor())

##
## End of file
##
