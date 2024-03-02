##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##
import numpy as np
from typing import List
from lib.node import Node


##
## Function to generate points
##
## This function create a random array of points in R^2
##
def generate_points(n: int) -> List[Node]:
    """
    Generates n random nodes in R^2.

    Args:
        n (int): Number of nodes to generate.

    Returns:
        List[Node]: A list of n nodes.
    """

    # Generate n points with x and y values between 0 and 1
    points: np.ndarray = np.random.randint(0, 101, size=(n, 2))

    # Create the nodes
    nodes: List[Node] = [Node(idx, x, y) for idx, (x, y) in enumerate(points)]

    # Return the nodes
    return nodes

    ##
    ## End of function
    ##


##
## This tests the generate_points function only if we're executing THIS current file.
##
## This is so that if we import the function from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    print(generate_points(10))

##
## End of file
##
