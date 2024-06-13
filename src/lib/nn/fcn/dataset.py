##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


##
## Imports
##
from torch.utils.data import Dataset
from lib.graph import Graph
from lib.utils.generate_graphs import generate_graphs
from lib.tsp.tspalgorithms import TSPAlgorithms
from lib.interfaces.tspalgorithm import TSPAlgorithm
from typing import List
import torch


##
## Graph dataset class
##
class GraphDataset(Dataset):
    ##
    ## Constructor
    ##
    def __init__(
        self, num_samples: int = 100, num_nodes: int = 7, node_features: int = 2
    ) -> None:
        """Initializes the GraphDataset class

        Args:
            num_samples (int): Number of samples to generate
            num_nodes (int): Number of nodes in each graph
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_features = node_features  ## x, y coordinates
        self.graphs: List[torch.Tensor] = []
        self.tours: List[torch.Tensor] = []

        self._graphs: List[Graph] = []

        self.set_rand()

        ##
        ## End of function
        ##

    ##
    ## Length of the dataset
    ##
    def __len__(self) -> int:
        """Length of the dataset

        Returns:
            int: The number of samples in the dataset
        """
        return self.num_samples

        ##
        ## End of function
        ##

    ##
    ## Get a sample from the dataset
    ##
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: A tuple containing the graph and the shortest tour
        """
        return torch.tensor(self.graphs[idx], dtype=torch.float32), torch.tensor(
            self.tours[idx], dtype=torch.float32
        )

        ##
        ## End of function
        ##

    ##
    ## Set the dataset to random graphs
    ##
    def set_rand(self) -> None:
        """Set the dataset to random graphs"""
        ##
        ## We'll save both the OOP graphs and the tensor graphs. We'll use the OOP graphs
        ## for generating a real tour using a prediction from our GNN, and the tensor graphs
        ## for actually training the GNN.
        ##

        ## Generate the OOP graphs
        self._graphs = generate_graphs(
            self.num_samples, self.num_nodes, self.num_features
        )

        ## Convert the graphs to tensors
        self.graphs = [graph.tensor() for graph in self._graphs]

        ## Set the shortest tours
        self.tours = [
            TSPAlgorithms.get_shortest_tour(graph, TSPAlgorithm.BruteForce).tensor()
            for graph in self._graphs
        ]

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the GraphDataset class only if we're executing THIS current file.
##
if __name__ == "__main__":  ##
    from torch.utils.data import DataLoader

    ## Create a dataset
    dataset = GraphDataset(num_samples=10, num_nodes=5)
    ## Create a dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    ## Iterate over the dataloader
    for i, (graph, tour) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print(f"Graph shape: {graph.shape}")
        print(f"Tour shape: {tour.shape}")
        print(f"Graph: {graph}")
        print(f"Tour: {tour}")
        print()

##
## End of file
##
