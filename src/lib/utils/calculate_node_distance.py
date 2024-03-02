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


##
## Calculate node distance
##
def calculate_node_distance(node1: Node, node2: Node) -> float:
    """Calculate the distance between two nodes

    Args:
        node1 (Node): The first node
        node2 (Node): The second node

    Returns:
        int: The distance between the nodes
    """
    # Get the x and y coordinates of the nodes
    x1, y1 = node1.x, node1.y
    x2, y2 = node2.x, node2.y

    # Calculate the distance between the nodes
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Return the distance
    return distance

    ##
    ## End of function
    ##


##
## Test the function
##
if __name__ == "__main__":
    # Create nodes
    node1 = Node(0, 0, 0)
    node2 = Node(1, 1, 1)

    # Calculate the distance
    distance = calculate_node_distance(node1, node2)

    # Print the distance
    print(distance)

##
## End of file
##
