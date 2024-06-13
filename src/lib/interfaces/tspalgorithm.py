##
## Set relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##


##
## TSPAlgorithm type
##
## TSPAlgorithm.BruteForce: Brute force algorithm
## TSPAlgorithm.GeneticAlgorithm: Genetic algorithm
## TSPAlgorithm.SimulatedAnnealing: Simulated annealing algorithm
## TSPAlgorithm.GreedyHeuristic: Greedy heuristic algorithm
## TSPAlgorithm.GNN: Graph neural network algorithm
## TSPAlgorithm.Opt2: 2-opt algorithm
## TSPAlgorithm.Opt3: 3-opt algorithm
## TSPAlgorithm.SimulatedAnnealing: Simulated annealing algorithm
##
class TSPAlgorithm:
    NoneType: str = "NoneType"
    BruteForce: str = "BruteForce"
    GeneticAlgorithm: str = "GeneticAlgorithm"
    SimulatedAnnealing: str = "SimulatedAnnealing"
    GreedyHeuristic: str = "GreedyHeuristic"
    GNN: str = "GNN"
    Opt2: str = "2-Opt"
    Opt3: str = "3-Opt"
    SimulatedAnnealing: str = "SimulatedAnnealing"

    ##
    ## End of class
    ##


##
## Test the TSPAlgorithm class
##
if __name__ == "__main__":
    print(TSPAlgorithm.NoneType)
    print(TSPAlgorithm.BruteForce)
    print(TSPAlgorithm.GeneticAlgorithm)
    print(TSPAlgorithm.SimulatedAnnealing)
    print(TSPAlgorithm.GreedyHeuristic)
    print(TSPAlgorithm.GNN)
    print(TSPAlgorithm.Opt2)
    print(TSPAlgorithm.Opt3)
    print(TSPAlgorithm.SimulatedAnnealing)

##
## End of file
##
