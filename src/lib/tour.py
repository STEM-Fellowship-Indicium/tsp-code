##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##
from lib.node import Node
from lib.types.tspalgorithm import TSPAlgorithm
from lib.utils.create_dist_matrix import create_dist_matrix
from typing import List
from torch import Tensor
import numpy as np
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
        distance: float = -1.0,
        algorithm: str = TSPAlgorithm.NoneType,
    ) -> None:
        """Initializer for the Tour class

        Args:
            nodes (List[Node], optional): The nodes of the tour. Defaults to [].
            distance (float, optional): The distance of the tour. Defaults to 0.
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
    ## Convert a prediction to a real tour
    ##
    ## TODO: Fix this function or the GNN. It will return nodes that are right beside eachother
    ## or on top of eachother.
    ##
    @staticmethod
    def from_prediction(real_node_positions: List[Node], pred: Tensor) -> "Tour":
        """Use a tour (predicted with the GNN) and it's node positions to generate a new tour with
        real nodes with real positions. The positions in the predicted tour are not real, they are
        just close to the real positions.

        Args:
            real_node_positions (List[List[float]]): The real node positions
            pred (Tensor): The predicted tour

        Returns:
            Tour: The tour with real nodes and real positions
        """
        ##
        ## Iterate over the nodes in the prediction. We'll get the real node with the closest
        ## position to the predicted node.
        ##
        nodes = [
            min(
                real_node_positions,
                key=lambda real_node: np.linalg.norm(
                    real_node.to_numpy() - node_tensor.numpy()
                ),
            )
            for node_tensor in pred.detach()
        ]

        ##
        ## Calculate the distance of the tour
        ##
        distance = 0
        distance_matrix = create_dist_matrix(nodes)
        for i in range(len(nodes) - 1):
            distance += distance_matrix[i, i + 1]

        ##
        ## Return the tour
        ##
        return Tour(nodes=nodes, distance=distance, algorithm=TSPAlgorithm.GNN)

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
    ## Convert the tour to a tensor
    ##
    def to_tensor(self) -> Tensor:
        """Convert the tour to a tensor

        Returns:
            Tensor: The tensor representation of the tour
        """
        return Tensor([node.to_numpy() for node in self.nodes])

        ##
        ## End of function
        ##

    ##
    ## Convert the tour to a numpy array
    ##
    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        """Convert the tour to a numpy array

        Returns:
            np.ndarray: The numpy representation of the tour
        """
        return np.array([node.to_numpy(dtype) for node in self.nodes])

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

    tour_json = tour.to_json()
    print(tour_json)

    tour_map = tour.to_map()
    print(tour_map)

    tour_tensor = tour.to_tensor()
    print(tour_tensor)

    tour_numpy = tour.to_numpy()
    print(tour_numpy)

    tour_from_map = Tour.from_map(tour_map)
    print(tour_from_map)

    tour_from_prediction = Tour.from_prediction(
        [Node(0, 0, 0), Node(1, 1, 1), Node(2, 2, 2)], tour_tensor
    )


##
## End of file
##
