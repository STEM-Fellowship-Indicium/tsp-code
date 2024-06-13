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
from lib.interfaces.tspalgorithm import TSPAlgorithm
from lib.utils.create_dist_matrix import create_dist_matrix
from typing import List, Union
from torch import Tensor
import numpy as np
import json, math


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
        return f"Tour(nodes={[str(node) for node in self.nodes]}, distance='{self.distance}', algorithm='{self.algorithm}')"

        ##
        ## End of function
        ##

    ##
    ## String representation of the tour
    ##
    def __repr__(self) -> str:
        """String representation of the tour

        Returns:
            str: The string representation of the tour
        """
        return self.__str__()

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
    def from_prediction(real_nodes: List[Node], pred: Tensor) -> "Tour":
        """Use a tour (predicted with the GNN) and it's node positions to generate a new tour with
        real nodes with real positions. The positions in the predicted tour are not real, they are
        just close to the real positions.

        Args:
            real_nodes (List[Node]): The real nodes
            pred (Tensor): The predicted tour

        Returns:
            Tour: The tour with real nodes and real positions
        """
        ##
        ## Detach the prediction
        ##
        pred = pred.detach()

        ##
        ## Iterate over the nodes in the prediction.
        ##
        ## We'll get the real node with the closest position to the predicted node.
        ##
        nodes = []

        for node in pred:
            closest_node = None
            closest_distance = math.inf

            for real_node in real_nodes:
                distance = math.sqrt(
                    (node[0] - real_node.x) ** 2 + (node[1] - real_node.y) ** 2
                )

                if distance < closest_distance and real_node not in nodes:
                    closest_node = real_node
                    closest_distance = distance

            nodes.append(closest_node)

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
    ## Normalize the tour
    ##
    def normalize(
        self, min: Union[float, int] = 0, max: Union[float, int] = 100
    ) -> "Tour":
        """Normalize the tour

        Args:
            min (List[float], optional): The minimum values for the tour. Defaults to [0, 0].
            max (List[float], optional): The maximum values for the tour. Defaults to [100, 100].

        Returns:
            Tour: The normalized tour
        """
        self.nodes = [node.normalize(min, max) for node in self.nodes]

        return self

        ##
        ## End of function
        ##

    ##
    ## Denormalize the tour
    ##
    def denormalize(
        self, min: Union[float, int] = 0, max: Union[float, int] = 100
    ) -> "Tour":
        """Denormalize the tour

        Args:
            min (List[float], optional): The minimum values for the tour. Defaults to [0, 0].
            max (List[float], optional): The maximum values for the tour. Defaults to [100, 100].

        Returns:
            Tour: The denormalized tour
        """
        self.nodes = [node.denormalize(min, max) for node in self.nodes]

        return self

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
    def to_tensor(self, dtype=np.float32) -> Tensor:
        """Convert the tour to a tensor

        Returns:
            Tensor: The tensor representation of the tour
        """
        nodes = [node for node in self.nodes]

        ## Convert nodes to np array
        nodes = np.array([node.to_numpy(dtype) for node in nodes])

        ## Return the tensor
        return Tensor(nodes)

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
    ## Create a copy of the tour
    ##
    def copy(self) -> "Tour":
        """Create a copy of the tour

        Returns:
            Tour: The copy of the tour
        """
        return Tour([node.copy() for node in self.nodes], self.distance, self.algorithm)

        ##
        ## End of function
        ##

    ##
    ## Print the tour
    ##
    def print(self) -> None:
        """Print the tour"""
        print(self.__str__())

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
