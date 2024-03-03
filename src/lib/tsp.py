##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


##
## Imports
##
from lib.graph import Graph
from lib.tour import Tour
from lib.types.tspalgorithm import TSPAlgorithm
from lib.utils.create_dist_matrix import create_dist_matrix
import itertools, math


##
## TSP type
##
## TSP.BruteForce: Brute force algorithm
## TSP.GeneticAlgorithm: Genetic algorithm
## TSP.SimulatedAnnealing: Simulated annealing algorithm
##
class TSP:
    ##
    ## Get the shortest tour
    ##
    @staticmethod
    def get_shortest_tour(graph: Graph, algorithm: TSPAlgorithm) -> Tour:
        """Get the shortest tour for the given algorithm

        Args:
            graph (Graph): The graph to solve
            algorithm (TSPAlgorithm): The algorithm to use

        Returns:
            Tour: The shortest tour
        """
        # Get the shortest tour based on the algorithm
        if algorithm == TSPAlgorithm.BruteForce:
            return TSP.brute_force(graph)
        elif algorithm == TSPAlgorithm.GeneticAlgorithm:
            return TSP.genetic(graph)
        elif algorithm == TSPAlgorithm.SimulatedAnnealing:
            return TSP.simulated_annealing(graph)

        # Return the shortest tour
        return Tour(algorithm=TSPAlgorithm.NoneType)

        ##
        ## End of function
        ##

    ##
    ## Brute force algorithm
    ##
    @staticmethod
    def brute_force(graph: Graph) -> Tour:
        """Brute force algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        # Store the shortest tour
        shortest_tour: Tour = Tour(algorithm=TSPAlgorithm.BruteForce)

        ##
        ## TODO: Implement the brute force algorithm
        ##
        distance_matrix = create_dist_matrix(graph.nodes)
        n = len(graph.nodes)
        shortest_distance = math.inf

        for path in itertools.permutations(range(n)):
            distance = 0
            for i in range(n - 1):
                distance += distance_matrix[path[i]][path[i + 1]]
            distance += distance_matrix[path[-1]][path[0]]

            if distance < shortest_distance:
                shortest_distance = distance
                shortest_tour.nodes = path

        shortest_tour.nodes = [graph.nodes[i] for i in shortest_tour.nodes]
        shortest_tour.distance = shortest_distance

        # Return the shortest tour
        return shortest_tour

        ##
        ## End of function
        ##

    ##
    ## Genetic algorithm
    ##
    @staticmethod
    def genetic(graph: Graph) -> Tour:
        """Genetic algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        # Store the shortest tour
        shortest_tour: Tour = Tour(algorithm=TSPAlgorithm.GeneticAlgorithm)

        ##
        ## TODO: Implement the genetic algorithm
        ##

        # Set the shortest tour nodes (the shortest path)
        shortest_tour.nodes = []

        # Return the shortest tour
        return shortest_tour

        ##
        ## End of function
        ##

    ##
    ## Simulated annealing algorithm
    ##
    @staticmethod
    def simulated_annealing(graph: Graph) -> Tour:
        """Simulated annealing algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        # Store the shortest tour
        shortest_tour: Tour = Tour(algorithm=TSPAlgorithm.SimulatedAnnealing)

        ##
        ## TODO: Implement the simulated annealing algorithm
        ##

        # Set the shortest tour nodes (the shortest path)
        shortest_tour.nodes = []

        # Return the shortest tour
        return shortest_tour

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the tsp class only if we're executing THIS current file.
##
## This is so that if we import the TSP class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    import time

    # Create a new graph
    graph = Graph.rand(10)

    # Solve the graph using the brute force algorithm
    start = time.time()
    res = TSP.brute_force(graph)
    speed = time.time() - start
    print(f"Brute force algorithm\nResult:{res}\nSpeed: {speed}s")

    # Save the graph and shortest tour to file (testing)
    graph.shortest_tour = res
    graph.export("data/tsp-test.json")

    # Solve the graph using the genetic algorithm
    start = time.time()
    res = TSP.genetic(graph)
    speed = time.time() - start
    print(f"\n\nGenetic algorithm\nResult:{res}\nSpeed: {speed}s")

    # Solve the graph using the simulated annealing algorithm
    start = time.time()
    res = TSP.simulated_annealing(graph)
    speed = time.time() - start
    print(f"\n\nSimulated annealing algorithm\nResult:{res}\nSpeed: {speed}s")


##
## End of file
##
