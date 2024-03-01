##
## Imports
##
from lib.utils import generate_points, calc_dist_pythagorean

##
## Execute the main function
##
if __name__ == "__main__":
    n = int(
        input("Enter the number of points to generate: ")
    )  # User inputs the number of points
    points = generate_points(n)  # Generate n random points

    print("Generated points:")
    for i, point in enumerate(points):
        print(f"Point {i}: (x={point[0]}, y={point[1]})")

    distances = calc_dist_pythagorean(points)  # Calculate distances between points
    print("\nDistances between points:")
    print(distances)
