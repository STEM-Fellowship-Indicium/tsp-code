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
from typing import Union
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
        return f"Node(idx='{self.idx}', x='{self.x}', y='{self.y}')"

        ##
        ## End of function
        ##

    ##
    ## String representation of the node
    ##
    def __repr__(self) -> str:
        """String representation of the node

        Returns:
            _type_: The string representation of the node
        """
        return self.__str__()

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
    def tensor(self, dtype=np.float32, normalize=(-1, -1)) -> Tensor:
        """Convert the node to a tensor

        Returns:
            Tensor: The tensor representation of the node
        """
        return Tensor(self.numpy(dtype, normalize))

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a numpy array
    ##
    def numpy(self, dtype=np.float32, normalize=(-1, -1)) -> np.ndarray:
        """Convert the node to a numpy array

        Returns:
            np.ndarray: The numpy array representation of the node
        """

        norm_min, norm_max = normalize

        if norm_min != -1 and norm_max != -1:
            self.normalize(norm_min, norm_max)

        np_array = np.array([self.x, self.y], dtype=dtype)

        if norm_min != -1 and norm_max != -1:
            self.denormalize(norm_min, norm_max)

        return np_array

        ##
        ## End of function
        ##

    ##
    ## Return a normalized version of the node
    ##
    ## This takes a node and normalizes it depending on min and max.
    ##
    ## min of 0 and max of 100 is just default since we're typically working
    ## with 0-100 coordinates.
    ##
    def normalize(
        self, min: Union[float, int] = 0, max: Union[float, int] = 100
    ) -> "Node":
        """Normalize the node"""
        self.x = (self.x - min) / (max - min)
        self.y = (self.y - min) / (max - min)

        return self

        ##
        ## End of function
        ##

    ##
    ## Denormalize the node
    ##
    def denormalize(
        self, min: Union[float, int] = 0, max: Union[float, int] = 100
    ) -> "Node":
        """Denormalize the node"""
        self.x = self.x * (max - min) + min
        self.y = self.y * (max - min) + min

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
        print(self.__str__())

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

    nump = node.numpy()
    print(nump)

    tens = node.tensor()
    print(tens)

    norm_prev = node.normalize()
    print(norm_prev, node)


##
## End of file
##
