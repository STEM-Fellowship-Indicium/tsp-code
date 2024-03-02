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
        shortest_tour: Tour = Tour(algorithm=TSPAlgorithm.SimulatedAnnealing)

        ##
        ## This is a placeholder.
        ##
        ## TODO: Implement the brute force algorithm
        ##
        for edge in graph.edges:
            print(edge)

        # Set the shortest tour nodes (the shortest path)
        shortest_tour.nodes = []

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
        shortest_tour: Tour = Tour(algorithm=TSPAlgorithm.SimulatedAnnealing)

        ##
        ## This is a placeholder.
        ##
        ## TODO: Implement the genetic algorithm
        ##
        for edge in graph.edges:
            print(edge)

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
        ## This is a placeholder.
        ##
        ## TODO: Implement the simulated annealing algorithm
        ##
        for edge in graph.edges:
            print(edge)

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
    # Create a new graph
    graph = Graph.rand(10)

    # Solve the graph using the brute force algorithm
    TSP.brute_force(graph)

    # Solve the graph using the genetic algorithm
    TSP.genetic(graph)

    # Solve the graph using the simulated annealing algorithm
    TSP.simulated_annealing(graph)


##
## End of file
##
