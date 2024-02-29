import numpy as np

# 1. Function to generate n random points in R^2
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


# 2. Function to calculate the distance between each pair of points
def calculate_distances_pythagorean(points):
    """
    calculates the distances using pythagorean formula

    Parameters:
        points (numpy.ndarray): An array of points in R^2.
    Returns:
    an array containing the distances between every two points
    """
    
    num_points = len(points)
    distances = np.zeros((num_points, num_points))  # Initialize a matrix to store distances
    
    for i in range(num_points):
        for j in range(i + 1, num_points):  # No need to calculate when j <= i, to avoid redundancy
            # Calculate the distance between points[i] and points[j]
            dist = np.sqrt((points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2)
            distances[i, j] = dist
            distances[j, i] = dist  # The distance matrix is symmetric
    
    return distances


# 3. Function to print details of any point
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


# Main part of the script
if __name__ == "__main__":
    n = int(input("Enter the number of points to generate: "))  # User inputs the number of points
    points = generate_points(n)  # Generate n random points

    print("Generated points:")
    for i, point in enumerate(points):
        print(f"Point {i}: (x={point[0]}, y={point[1]})")

    distances = calculate_distances_pythagorean(points)  # Calculate distances between points
    print("\nDistances between points:")
    print(distances)