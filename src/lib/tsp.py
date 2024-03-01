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


##
## TSP type
##
## TSP.BruteForce: Brute force algorithm
## TSP.GeneticAlgorithm: Genetic algorithm
## TSP.SimulatedAnnealing: Simulated annealing algorithm
##
class TSP:
    ##
    ## Brute force algorithm
    ##
    @staticmethod
    def brute_force(graph: Graph) -> None:
        """Brute force algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        for node in graph.nodes:
            print(node)

        return

        ##
        ## End of function
        ##

    ##
    ## Genetic algorithm
    ##
    @staticmethod
    def genetic(graph: Graph) -> None:
        """Genetic algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        for edge in graph.edges:
            print(edge)

        return

        ##
        ## End of function
        ##

    ##
    ## Simulated annealing algorithm
    ##
    @staticmethod
    def simulated_annealing(graph: Graph) -> None:
        """Simulated annealing algorithm for the TSP

        Args:
            graph (Graph): The graph to solve
        """
        for edge in graph.edges:
            print(edge)

        return

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## Execute the test
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
