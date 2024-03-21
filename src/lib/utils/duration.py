##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##
from typing import Any


##
## Duration calculate function
##
def duration(func: Any) -> None:
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"`{func.__name__}` took {end - start}s")
        return result

    return wrapper


##
## Test the function
##
if __name__ == "__main__":
    from lib.tsp.tspalgorithms import TSPAlgorithms
    from lib.graph import Graph

    graph = Graph.rand(num_nodes=7)

    @duration
    def brute_force(graph: Graph):
        return TSPAlgorithms.brute_force(graph)

    brute_force(graph)


##
## End of file
##
