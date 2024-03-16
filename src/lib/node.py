##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


##
## Imports
##
from torch import Tensor
from typing import List
import numpy as np
import json


##
## Node class
##
## This class represents a node in a graph.
##
class Node:
    ##
    ## Constructor
    ##
    def __init__(self, idx: int, x: float, y: float) -> None:
        """Initializer for the Node class

        Args:
            idx (int): The idx of the node
            x (int): The x coordinate of the node
            y (int): The y coordinate of the node
        """
        self.idx = idx
        self.x = x
        self.y = y

        ##
        ## End of function
        ##

    ##
    ## String representation of the node
    ##
    def __str__(self) -> str:
        """String representation of the node

        Returns:
            _type_: The string representation of the node
        """
        return f"({self.x}, {self.y})"

        ##
        ## End of function
        ##

    ##
    ## Node from map
    ##
    @staticmethod
    def from_map(map: dict = None) -> "Node":
        """Create a node from a map

        Args:
            map (dict): The map to create the node from

        Returns:
            Node: The node created from the map
        """
        if map is None:
            return None

        return Node(map["idx"], map["x"], map["y"])

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a map
    ##
    def to_map(self) -> dict:
        """Convert the node to a map

        Returns:
            dict: The map of the node
        """
        return {
            "idx": self.idx,
            "x": float(self.x),
            "y": float(self.y),
        }

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a json map
    ##
    def to_json(self) -> str:
        """Convert the node to a json map

        Returns:
            str: The json map of the node
        """
        return json.dumps(self.to_map(), indent=4)

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a tensor
    ##
    def to_tensor(self) -> Tensor:
        """Convert the node to a tensor

        Returns:
            Tensor: The tensor representation of the node
        """
        return Tensor([self.x, self.y])

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a numpy array
    ##
    def to_numpy(self, dtype=np.float32, normalize=True) -> np.ndarray:
        """Convert the node to a numpy array

        Returns:
            np.ndarray: The numpy array representation of the node
        """

        if normalize:
            self.normalize()

        np_array = np.array([self.x, self.y], dtype=dtype)

        if normalize:
            self.denormalize()

        return np_array

        ##
        ## End of function
        ##

    ##
    ## Return a normalized version of the node
    ##
    ## This takes a node and normalizes its x and y values to be between 0 and 1
    ##
    def normalize(
        self, min: List[float] = [0, 0], max: List[float] = [100, 100]
    ) -> "Node":
        """Normalize the node"""
        min_x, min_y = min
        max_x, max_y = max

        self.x = (self.x - min_x) / (max_x - min_x)
        self.y = (self.y - min_y) / (max_y - min_y)

        return self

        ##
        ## End of function
        ##

    ##
    ## Denormalize the node
    ##
    def denormalize(
        self, min: List[float] = [0, 0], max: List[float] = [100, 100]
    ) -> "Node":
        """Denormalize the node"""
        min_x, min_y = min
        max_x, max_y = max

        self.x = self.x * (max_x - min_x) + min_x
        self.y = self.y * (max_y - min_y) + min_y

        return self

        ##
        ## End of function
        ##

    ##
    ## Create a copy of the node
    ##
    def copy(self) -> "Node":
        """Create a copy of the node

        Returns:
            Node: The copy of the node
        """
        return Node(self.idx, self.x, self.y)

        ##
        ## End of function
        ##

    ##
    ## Print the node
    ##
    def print(self) -> None:
        """Print the node"""
        print(f"Node {self.idx}: ({self.x}, {self.y})")

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the node class only if we're executing THIS current file.
##
## This is so that if we import the Node class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    node = Node(0, 1, 2)
    print(node)
    print(node.to_json())
    node.print()

    nump = node.to_numpy()
    print(nump)

    tens = node.to_tensor()
    print(tens)

    norm_prev = node.normalize()
    print(norm_prev, node)


##
## End of file
##
