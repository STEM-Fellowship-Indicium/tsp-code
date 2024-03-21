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
from typing import List
from lib.utils.calculate_node_distance import calculate_node_distance

##
## Calculate tour distance
##
def calculate_tour_distance(nodes: List[Node]) -> float:
    """Calculates the total distance of the tour.

    Args:
        nodes (List[Node]): The nodes in the tour.

    Returns:
        float: The total distance of the tour.
    """
    distance = sum(calculate_node_distance(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes)))
    return distance

    ##
    ## End of function
    ##
    
    
##
## Test the function
##
if __name__ == "__main__":
    # Create nodes in a straight line
    nodes = [Node(idx=i, x=i*20, y=0) for i in range(5)]  # Nodes at (0,0), (10,0), (20,0), (30,0), (40,0)

    # Calculate the total distance of the tour
    total_distance = calculate_tour_distance(nodes)

    # Print the total distance
    print(f"Total distance of the tour: {total_distance}")

    
##
## End of file
##
