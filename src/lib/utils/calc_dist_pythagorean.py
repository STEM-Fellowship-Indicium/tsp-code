##
## Imports
##
import numpy as np


##
## Function to calculate distances using the pythagorean theorem
##
## This function calculates the pythagorean distance between every two points.
##
def calc_dist_pythagorean(points: np.ndarray) -> np.ndarray:
    """Calculate distances using the pythagorean theorem

    Args:
        points (np.ndarray): An array of points in R^2

    Returns:
        np.ndarray: A matrix of distances between every two points
    """

    # Store the number of points
    num_points = len(points)

    # Initialize a matrix to store distances
    distances = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i, num_points):
            # No need to calculate when j <= i, to avoid redundancy
            # Calculate the distance between points[i] and points[j]
            dist = np.sqrt(
                (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
            )

            distances[i, j] = dist
            distances[j, i] = dist  # The distance matrix is symmetric

    return distances

    ##
    ## End of function
    ##


##
## End of file
##
