##
## Imports
##
import numpy as np


##
## Function to generate points
##
## This function create a random array of points in R^2
##
def generate_points(n):
    """
    Generates n random points in R^2.

    Parameters:
    n (int): Number of points to generate.

    Returns:
    numpy.ndarray: An array of shape (n, 2) containing n points in R^2.
    """

    # Generate n points with x and y values between 0 and 1
    points = np.random.rand(n, 2)

    return points

    ##
    ## End of function
    ##


##
## End of file
##
