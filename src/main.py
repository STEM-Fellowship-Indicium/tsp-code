##
## Imports
##
from lib.graph import Graph
from lib.tsp.tspvisual import TSPVisual
from lib.utils.generate_graphs import generate_graphs
from lib.utils.import_graphs import import_graphs
from lib.types.tspalgorithm import TSPAlgorithm
from lib.tsp.tspalgorithms import TSPAlgorithms
from lib.utils.export_graphs import export_graphs

##
## Execute the main function
##
if __name__ == "__main__":
    ##
    ## Global variables to be manipulated by the user (from menu)
    ##
    graphs = None
    graph = None
    shortest_tour = None
    choice = None

    ##
    ## Generate graphs
    ##
    while choice != "-1":
        ##
        ## Menu for the user
        ##
        print("\nWhat would you like to do?")
        print("1. Generate and save random graphs")
        print("2. Generate a random graph")
        print("3. Visualize the graph and the shortest tour")
        print("4. Load graphs from a file")
        print("-1. Exit")
        choice = input("\nEnter the number of the option you would like to choose: ")

        ##
        ## 1. Generate and save random graphs
        ##
        if choice == "1":
            num_graphs = int(input("\nHow many graphs would you like to generate? "))
            num_nodes = int(
                input("How many nodes would you like to have in the graph? ")
            )

            algorithm_choice = input(
                "Which algorithm would you like to use?\n1. Brute Force\n2. Greedy Heuristic\n3. Two-Opt\n 4. Three-opt\n5. None\n\nYour choice: "
            )

            if algorithm_choice == "1":
                algorithm = TSPAlgorithm.BruteForce
            elif algorithm_choice == "2":
                algorithm = TSPAlgorithm.GreedyHeuristic
            elif algorithm_choice == "3":
                algorithm = TSPAlgorithm.Opt2
            elif algorithm_choice == "4":
                algorithm = TSPAlgorithm.Opt3
            else:
                algorithm = TSPAlgorithm.NoneType

            graphs = generate_graphs(
                n=num_graphs, num_nodes=num_nodes, algorithm=algorithm
            )

            for graph in graphs:
                print(f"Graph Shortest Tour: {graph.shortest_tour}")

            filename = input("\nEnter the filename to save the graphs to: ")

            export_graphs(graphs, filename)

            print(f"Graphs have been saved to `{filename}`")

        ##
        ## 2. Create a new graph
        ##
        elif choice == "2":
            num_nodes = int(
                input("\nHow many nodes would you like to have in the graph? ")
            )

            graph = Graph.rand(num_nodes=num_nodes)

            print(f"Graph generated! Shortest tour: {graph.shortest_tour}")

        ##
        ## 3. Visualize the graph and the shortest tour
        ##
        elif choice == "3":
            if not graph:
                print("Please create a graph first.\n")
                continue

            print("Which algorithms would you like to use?")
            print("1. Brute Force")
            print("2. Greedy Heuristic")
            print("3. Two-Opt")
            print("4. Three-Opt")

            algorithms = [
                algorithm
                for algorithm in input(
                    "Enter the numbers of the algorithms you would like to use (ex: 1,2,3,4): "
                )
                .strip()
                .split(",")
                if algorithm
            ]

            results = []

            for algorithm in algorithms:
                if algorithm == "1":
                    results.append(TSPAlgorithms.brute_force(graph))
                elif algorithm == "2":
                    results.append(TSPAlgorithms.greedy_heuristic(graph))
                elif algorithm == "3":
                    results.append(TSPAlgorithms.two_opt(graph))
                elif algorithm == "4":
                    results.append(TSPAlgorithms.three_opt(graph))

            graph.draw(results, ["red", "blue", "green"])

        ##
        ## 4. Load graphs from a file
        ##
        elif choice == "4":
            filename = input(
                "\nEnter the filename of the file with the cached graphs: "
            )

            graphs = import_graphs(filename)

            print(f"Graphs have been loaded from `{filename}`")

        ##
        ## Exit the program
        ##
        elif choice == "-1":
            break

        ##
        ## Invalid choice
        ##
        else:
            print("Invalid choice. Please try again.\n")

##
## End of file
##
