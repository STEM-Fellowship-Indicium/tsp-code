##
## Imports
##


##
## Function to print a point
##
def print_point(points, index):
    """
    Prints the details of a specified point.

    Parameters:
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
## End of file
##
