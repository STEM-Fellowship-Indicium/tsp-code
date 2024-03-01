##
## Imports
##
import numpy as np


##
## Function to print a point
##
def print_point(points: np.ndarray, index: int) -> None:
    """
    Prints the details of a specified point.

    Args:
        points (numpy.ndarray): An array of points in R^2.
        index (int): The index of the point to print.
    """

    if index < len(points):
        print(f"Point {index}: {points[index]}")
    else:
        print("Index is out of the range of generated points.")

    ##
    ## End of function
    ##


##
## This tests the print_point function only if we're executing THIS current file.
##
## This is so that if we import the function from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    print_point(np.array([[1, 2], [3, 4], [5, 6]]), 1)


##
## End of file
##
