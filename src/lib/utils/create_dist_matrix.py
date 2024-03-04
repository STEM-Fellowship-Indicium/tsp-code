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
from lib.utils.calculate_node_distance import calculate_node_distance


##
## Function to calculate distances using the pythagorean theorem
##
## This function calculates the pythagorean distance between every two points.
##
def create_dist_matrix(nodes: List[Node]) -> List[List[int]]:
    """Create a distance matrix for the given nodes

    Args:
        nodes (List[Node]): The nodes to create the distance matrix for

    Returns:
        List[List[int]]: The distance matrix
    """
    # Create the distance matrix
    dist_matrix = np.zeros((len(nodes), len(nodes)))

    # Iterate through the nodes
    for idx, node in enumerate(nodes):
        # Create a row for the node
        row = []

        # Iterate through the nodes again
        for other_node in nodes:
            # Get the distance between the nodes
            distance = calculate_node_distance(node, other_node)

            # Add the distance to the row
            row.append(distance)

        # Add the row to the distance matrix
        dist_matrix[idx] = row

    # Return the distance matrix
    return dist_matrix

    ##
    ## End of function
    ##


##
## Test the function
##
if __name__ == "__main__":
    # Create nodes
    nodes = [Node(i, i, i) for i in range(3)]

    # Create the distance matrix
    dist_matrix = create_dist_matrix(nodes)

    # Print the distance matrix
    print(dist_matrix)

##
## End of file
##
