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
from lib.utils.calculate_tour_distance import calculate_tour_distance

def three_opt_swap(nodes, i, j, k):
    """Tries all 3-opt swaps for a given triple of indices and returns the best swap.

    Args:
        nodes (List[Node]): The current tour.
        i, j, k (int): Indices in the tour to attempt to swap.

    Returns:
        tuple: The best new node order and its distance.
    """
    # Best configuration and its distance
    best_order = nodes
    best_distance = calculate_tour_distance(nodes)

    # Define segments A, B, C, D
    A, B, C, D = nodes[:i], nodes[i:j], nodes[j:k], nodes[k:]

    # Valid reconnections
    possibilities = [
        A + B + C + D,  # Original
        A + B + C[::-1] + D,  # Case 1: Reverse C
        A + B[::-1] + C + D,  # Reverse B
        A + B[::-1] + C[::-1] + D,  # Reverse B and C
        A + C + B + D,  # Swap B and C
        A + C + B[::-1] + D,  # Swap B and C, then reverse B
        A + C[::-1] + B + D,  # Swap B and C, then reverse C
        A + C[::-1] + B[::-1] + D,  # Swap B and C, reverse both
    ]

    for order in possibilities:
        current_distance = calculate_tour_distance(order)
        if current_distance < best_distance:
            best_order, best_distance = order, current_distance

    return best_order, best_distance
    ##
    ## End of function
    ##
    
##
## Test the function
##

if __name__ == "__main__":
    # Create nodes in a straight line
    nodes = [Node(idx=i, x=i*20, y=0) for i in range(5)]  # Nodes at (0,0), (10,0), (20,0), (30,0), (40,0)

    # Print the initial configuration and distance
    initial_distance = calculate_tour_distance(nodes)
    print(f"Initial distance of the tour: {initial_distance}")

    # Define indices for the 3-opt swap (choosing arbitrary indices within range)
    i, j, k = 1, 2, 4  # Example indices for swapping
    
    # Perform the 3-opt swap
    swapped_nodes, swapped_distance = three_opt_swap(nodes, i, j, k)

    # Print the result of the swap
    print(f"Distance after 3-opt swap: {swapped_distance}")
    for node in swapped_nodes:
        print(f"Node {node.idx}: ({node.x}, {node.y})")
        
    
##
## End of file
##
