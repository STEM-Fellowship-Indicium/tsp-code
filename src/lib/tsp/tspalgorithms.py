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
from lib.utils.calculate_tour_distance import calculate_tour_distance
from lib.utils.three_opt_swap import three_opt_swap
import itertools, math, random

from lib.utils.duration import duration


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
        ## Get the shortest tour based on the algorithm
        shortest_tour = TSPAlgorithms.get_shortest_tour(graph, algorithm)

        ## Set the shortest tour
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
        ## Get the shortest tour based on the algorithm
        if algorithm == TSPAlgorithm.BruteForce:
            return TSPAlgorithms.brute_force(graph)

        elif algorithm == TSPAlgorithm.GeneticAlgorithm:
            return TSPAlgorithms.genetic(graph)

        elif algorithm == TSPAlgorithm.SimulatedAnnealing:
            return TSPAlgorithms.simulated_annealing(graph)

        elif algorithm == TSPAlgorithm.GreedyHeuristic:
            return TSPAlgorithms.greedy_heuristic(graph)

        elif algorithm == TSPAlgorithm.Opt2:
            return TSPAlgorithms.two_opt(graph)

        ## Return the shortest tour
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
        ## Variables
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
            ## Add the distance from the last node to the first node
            ##
            distance += distance_matrix[graph.nodes.index(path[-1])][
                graph.nodes.index(path[0])
            ]

            ##
            ## If the distance is shorter than the current shortest distance,
            ## set the shortest distance to the current distance and set the
            ## shortest tour to the current path
            ##
            if distance < shortest_distance:
                shortest_distance = distance
                shortest_tour_nodes = path

        ## Return the shortest tour
        return Tour(
            nodes=shortest_tour_nodes,
            distance=shortest_distance,
            algorithm=TSPAlgorithm.BruteForce,
        )

        ##
        ## End of function
        ##

    ##
    ## 2-Opt algorithm
    ##
    @staticmethod
    def two_opt(graph: Graph, tour: Tour = None) -> Tour:
        """Applies the 2-opt algorithm to improve an initial tour and returns the improved tour as a Tour object.

        Args:
            graph (Graph): The initial graph with nodes representing the tour.
            tour (Tour): The initial tour to improve.

        Returns:
            Tour: An improved tour found using the 2-opt algorithm.
        """
        ##
        ## Initial setup
        ##
        ## We'll assume that the edges are connected in the order that the
        ## nodes are given.
        ##
        nodes = tour.nodes if tour is not None else graph.nodes
        best_distance = calculate_tour_distance(nodes)
        improved = True

        ##
        ## Our main loop
        ##
        while improved:
            improved = False

            for i in range(1, len(nodes) - 1):
                for j in range(i + 1, len(nodes)):
                    if j - i == 1:
                        continue  ## Skip adjacent nodes to avoid trivial swaps

                    new_nodes = nodes[:i] + nodes[i:j][::-1] + nodes[j:]
                    new_distance = calculate_tour_distance(new_nodes)

                    if new_distance < best_distance:
                        nodes = new_nodes  ## This becomes the new best tour
                        best_distance = new_distance
                        improved = True

        ## Assuming Tour class takes a list of Node objects, distance, and algorithm name
        return Tour(nodes=nodes, distance=best_distance, algorithm=TSPAlgorithm.Opt2)

        ##
        ## End of function
        ##

    ##
    ## Greedy Heuristic
    ##
    
    @staticmethod
    def three_opt(graph: Graph, tour: Tour = None) -> Tour:
        """Applies the 3-opt algorithm to improve an initial tour and returns the improved tour as a Tour object.

        Args:
            Graph: The initial graph with nodes representing the tour.
            Tour: The initial tour to improve.

        Returns:
            Tour: An improved tour found using the 3-opt algorithm.
        """
        nodes = tour.nodes if tour is not None else graph.nodes
        best_distance = calculate_tour_distance(nodes)
        improved = True
        
        while improved:
            improved = False
            n = len(nodes)

            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    for k in range(j + 2, n + (0 if i == 0 else 1)):
                        new_nodes, new_distance = three_opt_swap(nodes, i, j, k)

                        if new_distance < best_distance:
                            nodes, best_distance = new_nodes, new_distance
                            improved = True


        # Construct and return the improved tour
        return Tour(nodes=nodes, distance=best_distance, algorithm=TSPAlgorithm.Opt3)
        ##
        ## End of function
        ##

      
    ##
    ## Simulated annealing algorithm
    ##

    @staticmethod
    def simulated_annealing(graph: Graph, initial_tour: Tour = None) -> Tour:
        """Applies the Simulated Annealing algorithm to find an optimal tour for the TSP.

        Args:
            graph (Graph): The graph representing all the cities and distances between them.
            initial_tour (Tour): An optional initial tour from which to start the optimization.

        Returns:
            Tour: An optimized tour found using Simulated Annealing.
        """
        ##
        ## Initial setup
        ##
        current_nodes = initial_tour.nodes if initial_tour is not None else graph.nodes
        current_distance = calculate_tour_distance(current_nodes)
        best_nodes = current_nodes[:]
        best_distance = current_distance
        temperature = 195
        cooling_rate = 0.995

        ##
        ## Our main loop
        ##
        while temperature > 1:
            i, j = sorted(random.sample(range(len(current_nodes)), 2))
            new_nodes = (
                current_nodes[:i] +
                current_nodes[i:j][::-1] +
                current_nodes[j:]
            )
            new_distance = calculate_tour_distance(new_nodes)

            if (new_distance < current_distance or
                math.exp((current_distance - new_distance) / temperature) > random.random()):
                current_nodes = new_nodes
                current_distance = new_distance

                if new_distance < best_distance:
                    best_nodes = new_nodes[:]
                    best_distance = new_distance

            temperature *= cooling_rate

        ## Assuming Tour class takes a list of Node objects, distance, and algorithm name
        return Tour(nodes=best_nodes, distance=best_distance, algorithm=TSPAlgorithm.SimulatedAnnealing)

        ##
        ## End of function
        ##
    

    @staticmethod
    def greedy_heuristic(graph: Graph) -> Tour:
        """Greedy heuristic algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        ## Variables
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
## This tests the tsp class only if we're executing THIS current file.
##
## This is so that if we import the TSP class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    import time

    ## Create a new graph
    graph = Graph.rand(num_nodes=10)

    ##
    ## Track the duration of each algorithm
    ##
    @duration
    def test_brute_force(graph: Graph) -> Tour:
        return TSPAlgorithms.brute_force(graph)

    @duration
    def test_genetic(graph: Graph) -> Tour:
        return TSPAlgorithms.genetic(graph)

    @duration
    def test_simulated_annealing(graph: Graph) -> Tour:
        return TSPAlgorithms.simulated_annealing(graph)

    @duration
    def test_greedy_heuristic(graph: Graph) -> Tour:
        return TSPAlgorithms.greedy_heuristic(graph)

    @duration
    def test_two_opt(graph: Graph) -> Tour:
        return TSPAlgorithms.two_opt(graph)

    brute_force_res = test_brute_force(graph)
    genetic_res = test_genetic(graph)
    simulated_annealing_res = test_simulated_annealing(graph)
    greedy_heuristic_res = test_greedy_heuristic(graph)
    two_opt_res = test_two_opt(graph)

    graph.draw(
        [
            brute_force_res,
            genetic_res,
            simulated_annealing_res,
            greedy_heuristic_res,
            two_opt_res,
        ],
        ["red", "green", "blue", "yellow", "purple"],
    )

    """
    Output:
    
    `test_brute_force` took 10.295390844345093s
    `test_genetic` took 5.9604644775390625e-06s
    `test_simulated_annealing` took 1.6689300537109375e-06s
    `test_greedy_heuristic` took 0.00022912025451660156s
    `test_two_opt` took 0.0005450248718261719s
    """


##
## End of file
##
