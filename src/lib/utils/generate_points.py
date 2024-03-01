##
## Imports
##
import numpy as np


##
## Function to generate points
##
## This function create a random array of points in R^2
##
def generate_points(n: int) -> np.ndarray:
    """
    Generates n random points in R^2.

    Args:
        n (int): Number of points to generate.

    Returns:
        numpy.ndarray: An array of shape (n, 2) containing n points in R^2.
    """

    # Generate n points with x and y values between 0 and 1
    points = np.random.randint(0, 101, size=(n, 2))

    return points

    ##
    ## End of function
    ##


##
## End of file
##
