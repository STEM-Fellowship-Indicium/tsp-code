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
import itertools
import matplotlib.pyplot as plt


##
## TSP Visualizations
##
class TSPVisual:
    ##
    ## Brute force algorithm with visualization
    ##
    ## Show the brute force algorithm with visualization in real time
    ##
    @staticmethod
    def brute_force(graph: Graph) -> None:
        # Variables
        n = len(graph.nodes)

        # Draw the graph nodes
        plt.figure()
        for node in graph.nodes:
            plt.plot(node.x, node.y, "o", color="blue")

        ## Draw the edges that are being checked
        ##
        ## We also need to clear the plot and draw the nodes again
        ## so that we can see the edges being checked in real time.
        ##
        for path in itertools.permutations(range(n)):
            plt.clf()  # Clear the plot

            # Draw the graph nodes
            for node in graph.nodes:
                plt.plot(node.x, node.y, "o", color="blue")

            # Draw the edges
            for i in range(n - 1):
                plt.plot(
                    [graph.nodes[path[i]].x, graph.nodes[path[i + 1]].x],
                    [graph.nodes[path[i]].y, graph.nodes[path[i + 1]].y],
                    color="black",
                )

            # Draw the last edge
            plt.plot(
                [graph.nodes[path[-1]].x, graph.nodes[path[0]].x],
                [graph.nodes[path[-1]].y, graph.nodes[path[0]].y],
                color="black",
            )

            # Pause for a moment so that we can actually see the edges being checked
            plt.pause(0.001)

        # Show the graph
        plt.show()

    ##
    ## Greedy heuristic algorithm with visualization
    ##
    @staticmethod
    def greedy_heuristic(graph: Graph) -> None:
        """Visualize the greedy heuristic algorithm for the TSP

        Args:
            graph (Graph): The graph to solve

        Returns:
            None
        """

        return

        ##
        ## End of function
        ##

    ##
    ## Genetic algorithm visualization
    ##
    @staticmethod
    def genetic(graph: Graph) -> None:
        """Visualize the genetic algorithm for the TSP

        Args:
            graph (Graph): The graph to solve

        Returns:
            None
        """

        return

        ##
        ## End of function
        ##

    ##
    ## Simulated annealing algorithm visualization
    ##
    @staticmethod
    def simulated_annealing(graph: Graph) -> None:
        """Visualize the simulated annealing algorithm for the TSP

        Args:
            graph (Graph): The graph to solve

        Returns:
            None
        """
        ##
        ## TODO: Implement the simulated annealing algorithm visualization
        ##

        return

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the TSPVisual class only if we're executing THIS current file.
##
## This is so that if we import the TSPVisual class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    # Create a new graph
    graph = Graph.rand(7)

    # Greedy heuristic visual
    TSPVisual.greedy_heuristic(graph)

    # Brute force visual
    TSPVisual.brute_force(graph)


##
## End of file
##
