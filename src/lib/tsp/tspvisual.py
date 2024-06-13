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
import matplotlib.pyplot as plt
from lib.tour import Tour
from lib.utils.create_dist_matrix import create_dist_matrix
import itertools, math, random
from lib.utils.calculate_tour_distance import calculate_tour_distance
from lib.utils.three_opt_swap import three_opt_swap


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
        ## Variables
        n = len(graph.nodes)

        ## Draw the graph nodes
        plt.figure()
        for node in graph.nodes:
            plt.plot(node.x, node.y, "o", color="blue")

        ## Draw the edges that are being checked
        ##
        ## We also need to clear the plot and draw the nodes again
        ## so that we can see the edges being checked in real time.
        ##
        for path in itertools.permutations(range(n)):
            plt.clf()  ## Clear the plot

            ## Draw the graph nodes
            for node in graph.nodes:
                plt.plot(node.x, node.y, "o", color="blue")

            ## Draw the edges
            for i in range(n - 1):
                plt.plot(
                    [graph.nodes[path[i]].x, graph.nodes[path[i + 1]].x],
                    [graph.nodes[path[i]].y, graph.nodes[path[i + 1]].y],
                    color="black",
                )

            ## Draw the last edge
            plt.plot(
                [graph.nodes[path[-1]].x, graph.nodes[path[0]].x],
                [graph.nodes[path[-1]].y, graph.nodes[path[0]].y],
                color="black",
            )

            ## Pause for a moment so that we can actually see the edges being checked
            plt.pause(0.001)

        ## Show the graph
        plt.show()

    ##
    ## Two-opt algorithm with visualization
    ##
    @staticmethod
    def two_opt(graph: Graph, tour: Tour = None) -> None:
        """Visualizes the 2-opt algorithm in real-time as it improves the tour.

        Args:
            graph (Graph): The graph to solve
            tour (Tour): The tour to improve
        """
        ##
        ## We'll assume that the edges are connected in the order that the
        ## nodes are given.
        ##
        nodes = tour.nodes if tour is not None else graph.nodes

        ##
        ## Enable interactive plotting and create a new figure and axis
        ##
        plt.ion()
        _, ax = plt.subplots()

        ##
        ## Initial drawing of the graph nodes
        ##
        ax.scatter([node.x for node in nodes], [node.y for node in nodes], color="blue")
        plt.draw()

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
                        nodes[:] = new_nodes  ## Update the nodes list in-place
                        best_distance = new_distance
                        improved = True

                        ## Clear the plot for the next drawing
                        ax.clear()

                        ## Redraw the graph nodes and edges
                        ax.scatter(
                            [node.x for node in nodes],
                            [node.y for node in nodes],
                            color="blue",
                        )

                        ##
                        ## Redraw the tour edges
                        ##
                        for k in range(len(nodes)):
                            next_k = (k + 1) % len(nodes)
                            ax.plot(
                                [nodes[k].x, nodes[next_k].x],
                                [nodes[k].y, nodes[next_k].y],
                                color="black",
                            )

                        plt.draw()
                        plt.pause(0.01)

        plt.ioff()  ## Turn off interactive mode
        plt.show()

    ##
    ## Three-opt algorithm with visualization
    ##
    @staticmethod
    def three_opt(graph: Graph, tour: Tour = None) -> None:
        nodes = tour.nodes if tour is not None else graph.nodes

        plt.ion()
        fig, ax = plt.subplots()

        # Initial tour drawing
        ax.scatter([node.x for node in nodes], [node.y for node in nodes], color="blue")
        for i in range(len(nodes)):
            next_i = (i + 1) % len(nodes)
            ax.plot(
                [nodes[i].x, nodes[next_i].x],
                [nodes[i].y, nodes[next_i].y],
                color="black",
            )
        plt.draw()
        plt.pause(2)  # Pause to show initial configuration

        best_distance = calculate_tour_distance(nodes)
        improved = True

        while improved:
            improved = False
            for i in range(1, len(nodes) - 2):
                for j in range(i + 2, len(nodes) - 1):
                    for k in range(j + 2, len(nodes) + (0 if i > 0 else 1)):
                        new_nodes, new_distance = three_opt_swap(nodes, i, j, k)
                        if new_distance < best_distance:
                            nodes[:] = new_nodes
                            best_distance = new_distance
                            improved = True

                            # Visualization update
                            ax.clear()
                            ax.scatter(
                                [node.x for node in nodes],
                                [node.y for node in nodes],
                                color="blue",
                            )
                            for l in range(len(nodes)):
                                next_l = (l + 1) % len(nodes)
                                ax.plot(
                                    [nodes[l].x, nodes[next_l].x],
                                    [nodes[l].y, nodes[next_l].y],
                                    color="black",
                                )
                            plt.draw()
                            plt.pause(0.5)

        plt.ioff()
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
        ##
        ## Plot all the nodes and edges
        ##
        plt.figure()

        ##
        ## Draw the nodes
        ##
        for node in graph.nodes:
            plt.plot(node.x, node.y, "o", color="blue")

        ##
        ## Apply the greedy heuristic algorithm
        ##
        n: int = len(graph.nodes)
        shortest_tour_nodes = [graph.nodes[0]]
        distance_matrix = create_dist_matrix(graph.nodes)

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
                ##
                ## Get the closest node
                ##
                if graph.nodes[j] not in shortest_tour_nodes:
                    ##
                    ## Draw a line from the current node to the node being checked
                    ##
                    plt.plot(
                        [current_node.x, graph.nodes[j].x],
                        [current_node.y, graph.nodes[j].y],
                        color="green",
                    )

                    plt.pause(0.1)

                    ##
                    ## Remove the line from the current node to the node being checked
                    ##
                    plt.plot(
                        [current_node.x, graph.nodes[j].x],
                        [current_node.y, graph.nodes[j].y],
                        color="white",
                    )

                    distance = distance_matrix[graph.nodes.index(current_node)][j]

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_node = graph.nodes[j]

            ##
            ## Add the closest node to the shortest tour and plot it
            ##
            shortest_tour_nodes.append(closest_node)

            ##
            ## Draw the shortest tour
            ##
            for i in range(len(shortest_tour_nodes) - 1):
                plt.plot(
                    [shortest_tour_nodes[i].x, shortest_tour_nodes[i + 1].x],
                    [shortest_tour_nodes[i].y, shortest_tour_nodes[i + 1].y],
                    color="red",
                )

        ##
        ## Draw the last edge
        ##
        plt.plot(
            [shortest_tour_nodes[-1].x, shortest_tour_nodes[0].x],
            [shortest_tour_nodes[-1].y, shortest_tour_nodes[0].y],
            color="red",
        )

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
    ## Simulated Annealing algorithm with visualization
    ##
    @staticmethod
    def simulated_annealing(graph: Graph, initial_tour: Tour = None) -> None:
        """Applies the Simulated Annealing algorithm with real-time visualization to find an optimal tour for the TSP.

        Args:
            graph (Graph): The graph representing all the cities and distances between them.
            initial_tour (Tour): An optional initial tour from which to start the optimization.
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
        ## Enable interactive plotting and create a new figure and axis
        ##
        plt.ion()
        _, ax = plt.subplots()

        ##
        ## Initial drawing of the graph nodes
        ##
        ax.scatter(
            [node.x for node in current_nodes],
            [node.y for node in current_nodes],
            color="blue",
        )
        # Draw initial tour
        for i in range(len(current_nodes)):
            next_i = (i + 1) % len(current_nodes)
            ax.plot(
                [current_nodes[i].x, current_nodes[next_i].x],
                [current_nodes[i].y, current_nodes[next_i].y],
                color="black",
            )
        plt.draw()

        ##
        ## Our main loop
        ##
        while temperature > 1:
            i, j = sorted(random.sample(range(len(current_nodes)), 2))
            new_nodes = current_nodes[:i] + current_nodes[i:j][::-1] + current_nodes[j:]
            new_distance = calculate_tour_distance(new_nodes)

            if (
                new_distance < current_distance
                or math.exp((current_distance - new_distance) / temperature)
                > random.random()
            ):
                current_nodes = new_nodes
                current_distance = new_distance

                if new_distance < best_distance:
                    best_nodes = new_nodes[:]
                    best_distance = new_distance

                ## Clear the plot for the next drawing
                ax.clear()

                ## Redraw the graph nodes
                ax.scatter(
                    [node.x for node in current_nodes],
                    [node.y for node in current_nodes],
                    color="blue",
                )

                ## Redraw the tour edges
                for k in range(len(current_nodes)):
                    next_k = (k + 1) % len(current_nodes)
                    ax.plot(
                        [current_nodes[k].x, current_nodes[next_k].x],
                        [current_nodes[k].y, current_nodes[next_k].y],
                        color="black",
                    )

                plt.draw()
                plt.pause(0.05)  # Slight pause to visualize updates

            temperature *= cooling_rate

        plt.ioff()  ## Turn off interactive mode
        plt.show()

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
    ## Create a new graph
    graph = Graph.rand(7)

    ## Greedy heuristic visual
    print("Greedy heuristic visual")
    TSPVisual.greedy_heuristic(graph)
    plt.pause(5)

    ## Brute force visual
    print("Brute force visual")
    TSPVisual.brute_force(graph)
    plt.pause(5)

    ## Two-opt visual
    print("Two-opt visual")
    TSPVisual.two_opt(graph)
    plt.pause(5)

    ## Three-opt visual
    print("Three-opt visual")
    TSPVisual.three_opt(graph)
    plt.pause(5)

    ## Simulated Annealing visual
    print("Simulated Annealing visual")
    TSPVisual.simulated_annealing(graph)
    plt.pause(5)


##
## End of file
##
