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
        # Variables
        distance_matrix = create_dist_matrix(graph.nodes)
        shortest_distance = math.inf
        shortest_tour_nodes = []

        ##
        ## Generate the paths (node indices) use factorial since there are n! paths
        ## and we don't want indices next to each other to be the same.
        ##
        paths = [
            list(path) for path in itertools.permutations(graph.nodes, len(graph.nodes))
        ]

        ##
        ## Find the shortest tour
        ##
        for path in paths:
            ##
            ## Calculate the distance of the path
            ##
            distance = 0
            for i in range(len(path) - 1):
                distance += distance_matrix[graph.nodes.index(path[i])][
                    graph.nodes.index(path[i + 1])
                ]

            ##
            ## If the distance is shorter than the current shortest distance,
            ## set the shortest distance to the current distance and set the
            ## shortest tour to the current path
            ##
            if distance < shortest_distance:
                shortest_distance = distance
                shortest_tour_nodes = path

        # Return the shortest tour
        return Tour(
            nodes=shortest_tour_nodes,
            distance=shortest_distance,
            algorithm=TSPAlgorithm.BruteForce,
        )

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
        # Variables
        distance_matrix = create_dist_matrix(graph.nodes)
        shortest_distance = 0
        shortest_tour_nodes = [graph.nodes[0]]

        ##
        ## Find the shortest tour. This is done by starting at the first node,
        ## then going to the closest node, then going to the closest node from
        ## that node, and so on.
        ##
        ## Since this is a heuristic, it may not always find the shortest tour.
        ##
        n: int = len(graph.nodes)
        for _ in range(n - 1):
            ##
            ## Get the current node
            ##
            current_node = shortest_tour_nodes[-1]

            ##
            ## Get the closest node
            ##
            closest_node = None
            closest_distance = math.inf

            for j in range(n):
                if graph.nodes[j] not in shortest_tour_nodes:
                    distance = distance_matrix[graph.nodes.index(current_node)][j]

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_node = graph.nodes[j]

            ##
            ## Add the closest node to the shortest tour
            ##
            shortest_tour_nodes.append(closest_node)
            shortest_distance += closest_distance

        ##
        ## Return the shortest tour
        ##
        return Tour(
            nodes=shortest_tour_nodes,
            distance=shortest_distance,
            algorithm=TSPAlgorithm.GreedyHeuristic,
        )

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
    # graph.export("data/graph-tsp.json")

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
