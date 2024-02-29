##
## Imports
##
import numpy as np


##
## Function to calculate distances using Pythagorean theorem
##
## This function calculates the distances between every two points using the Pythagorean theorem.
##
def calculate_distances_pythagorean(points):
    """
    calculates the distances using pythagorean formula

    Parameters:
        points (numpy.ndarray): An array of points in R^2.
    Returns:
    an array containing the distances between every two points
    """

    num_points = len(points)
    distances = np.zeros(
        (num_points, num_points)
    )  # Initialize a matrix to store distances

    for i in range(num_points):
        for j in range(
            i + 1, num_points
        ):  # No need to calculate when j <= i, to avoid redundancy
            # Calculate the distance between points[i] and points[j]
            dist = np.sqrt(
                (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
            )
            distances[i, j] = dist
            distances[j, i] = dist  # The distance matrix is symmetric

    return distances
