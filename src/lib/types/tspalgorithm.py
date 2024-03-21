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
##
class TSPAlgorithm:
    NoneType: str = "NoneType"
    BruteForce: str = "BruteForce"
    GeneticAlgorithm: str = "GeneticAlgorithm"
    SimulatedAnnealing: str = "SimulatedAnnealing"
    GreedyHeuristic: str = "GreedyHeuristic"
    GNN: str = "GNN"
    Opt2: str = "2-Opt"

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

##
## End of file
##
