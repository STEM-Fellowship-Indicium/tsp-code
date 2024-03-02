##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


##
## Imports
##
import torch
from torch.utils.data import Dataset
import numpy as np


##
## Graph dataset class
##
class GraphDataset(Dataset):
    ##
    ## Constructor
    ##
    def __init__(
        self, num_samples: int = 10, num_nodes: int = 10, node_features: int = 2
    ) -> None:
        """Initializes the GraphDataset class

        Args:
            num_samples (int): Number of samples to generate
            num_nodes (int): Number of nodes in each graph
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.num_features = node_features  # x, y coordinates
        self.graphs = []
        self.tours = []

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
            self.tours[idx], dtype=torch.int64
        )

        ##
        ## End of function
        ##

    ##
    ## Set the dataset to random graphs
    ##
    def set_rand(self) -> None:
        """Set the dataset to random graphs"""
        for _ in range(self.num_samples):
            nodes = torch.rand((self.num_nodes, self.num_features), dtype=torch.float32)
            dist = self.dist(nodes)
            tour = self.nearest_neighbor(dist)

            self.graphs.append(nodes)
            self.tours.append(tour)

        ##
        ## End of function
        ##

    ##
    ## Calculate the distance matrix
    ##
    def dist(self, nodes: torch.Tensor) -> torch.Tensor:
        """Calculate the distance matrix

        Args:
            nodes (torch.Tensor): The nodes

        Returns:
            torch.Tensor: The distance matrix
        """
        # Get the x and y coordinates of the nodes
        x, y = nodes[:, 0], nodes[:, 1]
        # Calculate the distance matrix
        return torch.sqrt((x.unsqueeze(1) - x) ** 2 + (y.unsqueeze(1) - y) ** 2)

        ##
        ## End of function

    ##
    ## Nearest neighbor algorithm
    ##
    def nearest_neighbor(self, dist: torch.Tensor) -> torch.Tensor:
        """Nearest neighbor algorithm

        Args:
            dist (torch.Tensor): The distance matrix

        Returns:
            torch.Tensor: The shortest tour
        """
        n = dist.size(0)
        tour = [0]
        visited = {0}
        for _ in range(n - 1):
            last = tour[-1]
            nearest = min(
                (i for i in range(n) if i not in visited), key=lambda x: dist[last, x]
            )
            tour.append(nearest)
            visited.add(nearest)
        return torch.tensor(tour)

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the GraphDataset class only if we're executing THIS current file.
##
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Create a dataset
    dataset = GraphDataset(num_samples=10, num_nodes=20)
    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # Iterate over the dataloader
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
