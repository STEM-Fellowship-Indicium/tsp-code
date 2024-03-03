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
import itertools, math, random


##
## TSP Algorithms
##
class TSPAlgorithms:
    ##
    ## Set the shortest tour
    ##
    @staticmethod
    def set_shortest_tour(graph: Graph, algorithm: str) -> None:
        """Set the shortest tour for the given algorithm

        Args:
            graph (Graph): The graph to solve
            algorithm (TSPAlgorithm): The algorithm to use
        """
        # Get the shortest tour based on the algorithm
        shortest_tour = TSPAlgorithms.get_shortest_tour(graph, algorithm)

        # Set the shortest tour
        graph.shortest_tour = shortest_tour

        ##
        ## End of function
        ##

    ##
    ## Get the shortest tour
    ##
    @staticmethod
    def get_shortest_tour(graph: Graph, algorithm: str) -> Tour:
        """Get the shortest tour for the given algorithm

        Args:
            graph (Graph): The graph to solve
            algorithm (TSPAlgorithm): The algorithm to use

        Returns:
            Tour: The shortest tour
        """
        # Get the shortest tour based on the algorithm
        if algorithm == TSPAlgorithm.BruteForce:
            return TSPAlgorithms.brute_force(graph)

        elif algorithm == TSPAlgorithm.GeneticAlgorithm:
            return TSPAlgorithms.genetic(graph)

        elif algorithm == TSPAlgorithm.SimulatedAnnealing:
            return TSPAlgorithms.simulated_annealing(graph)

        elif algorithm == TSPAlgorithm.GreedyHeuristic:
            return TSPAlgorithms.greedy_heuristic(graph)

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

        # Variables
        distance_matrix = create_dist_matrix(graph.nodes)
        n = len(graph.nodes)
        shortest_distance = math.inf

        ## Find the shortest tour
        ##
        ## itertools.permutations returns all possible permutations of the nodes.
        ## For example: [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ##
        ## https://www.geeksforgeeks.org/python-itertools-permutations/
        ##
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
    ## Greedy Heuristic
    ##
    @staticmethod
    def greedy_heuristic(graph: Graph) -> Tour:
        """Greedy heuristic algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        # Store the shortest tour
        shortest_tour: Tour = Tour(algorithm=TSPAlgorithm.GreedyHeuristic)

        # Variables
        distance_matrix = create_dist_matrix(graph.nodes)
        n = len(graph.nodes)
        visited = [False] * n
        shortest_distance = 0
        shortest_tour.nodes = [graph.nodes[0]]

        # Find the shortest tour
        for _ in range(n - 1):
            current_node = shortest_tour.nodes[-1]
            visited[graph.nodes.index(current_node)] = True
            min_distance = math.inf
            next_node = None

            for i in range(n):
                if not visited[i]:
                    if (
                        distance_matrix[graph.nodes.index(current_node)][i]
                        < min_distance
                    ):
                        min_distance = distance_matrix[graph.nodes.index(current_node)][
                            i
                        ]
                        next_node = graph.nodes[i]

            shortest_tour.nodes.append(next_node)
            shortest_distance += min_distance

        shortest_distance += distance_matrix[
            graph.nodes.index(shortest_tour.nodes[-1])
        ][0]
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

        return

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

        return

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
    brute_res = TSPAlgorithms.brute_force(graph)
    speed = time.time() - start
    print(f"Brute force algorithm\nResult:{brute_res}\nSpeed: {speed}s")

    # Save the graph and shortest tour to file (testing)
    # graph.shortest_tour = res
    # graph.export("data/tsp-test.json")

    # Solve the graph using the genetic algorithm
    start = time.time()
    gen_res = TSPAlgorithms.genetic(graph)
    speed = time.time() - start
    print(f"\n\nGenetic algorithm\nResult:{gen_res}\nSpeed: {speed}s")

    # Solve the graph using the simulated annealing algorithm
    start = time.time()
    sim_anneal_res = TSPAlgorithms.simulated_annealing(graph)
    speed = time.time() - start
    print(
        f"\n\nSimulated annealing algorithm\nResult:{sim_anneal_res}\nSpeed: {speed}s"
    )

    # Solve the graph using the greedy heuristic algorithm
    start = time.time()
    greedy_res = TSPAlgorithms.greedy_heuristic(graph)
    speed = time.time() - start
    print(f"\n\nGreedy heuristic algorithm\nResult:{greedy_res}\nSpeed: {speed}s")

    # Plot the graph and tours
    graph.draw(tours=[brute_res, greedy_res], colors=["yellow", "red"])


##
## End of file
##
