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
import torch.nn as nn
import torch.optim as optim


##
## GNN class
##
class GNN(nn.Module):
    ##
    ## Constructor
    ##
    def __init__(
        self, node_features: int = 2, hidden_dim: int = 16, output_dim: int = 2
    ) -> None:
        """Initializer for the GNN class

        Args:
            node_features (int): The number of features for each node
            hidden_dim (int): The hidden dimension for the GNN
            output_dim (int): The output dimension for the GNN
        """
        super(GNN, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ##
        ## Criterion and optimizer
        ##
        self.criterion = nn.MSELoss
        self.optimizer = optim.Adam

        ##
        ## Create the sequence of layers
        ##
        ## Args for: nn.Linear(input_channels, output_channels)
        ##
        self.sequence = nn.Sequential(
            ## Layer 1: Input
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            ## Layer 2: Hidden
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            ## Layer 3: Hidden
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            ## Layer 4: Hidden
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            ## Layer 5: Hidden
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            ## Layer 6: Output
            nn.Linear(hidden_dim, output_dim),
        )

        ##
        ## End of function
        ##

    ##
    ## Forward pass
    ##
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GNN

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The output of the GNN

        """
        ##
        ## We want the output to be positions of the nodes. We'll then use these
        ## distances to locate the nearest node to each node.
        ##
        ## We'll take the output of the GNN and each row will be the x and y
        ## coordinates of a node.
        ##
        ## Perform the forward pass.
        ##
        x = self.sequence(x)

        ##
        ## Reshape the output. We want the output to be of shape (num_nodes, self.output_dim) [(rows, columns)]
        ##
        ## Note that: -1 automatically calculates the number of rows needed.
        ##
        # x = x.view(-1, self.output_dim)

        ##
        ## Return the output
        ##
        return x

    ##
    ## End of class
    ##


##
## This tests the gnn class only if we're executing THIS current file.
##
## This is so that if we import the GNN class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    ##
    ## Test Imports
    ##
    from graphdataset import GraphDataset
    from lib.tour import Tour
    from lib.tsp.tspalgorithms import TSPAlgorithms, TSPAlgorithm

    ##
    ## If our device has a GPU, use it (for speed).
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##
    ## Create a dataset using the custom GraphDataset class
    ##
    dataset = GraphDataset(num_samples=10, num_nodes=5, node_features=2)

    ##
    ## Create the GNN model
    ##
    model = GNN(node_features=2, hidden_dim=16, output_dim=2).to(device)

    ##
    ## Create a loss function and an optimizer
    ##
    ## The MSELoss function is most accurate for the TSP
    ## The optimizer most accurate for the TSP is the Adam optimizer
    ##
    criterion = model.criterion()  ## or nn.MSELoss()
    optimizer = model.optimizer(
        model.parameters(), lr=0.001
    )  ## or optim.Adam(model.parameters(), lr=0.001)

    ##
    ## Train the model with 1000 epochs
    ##
    epochs: int = 1000
    ##
    ## Training loop
    ##
    for epoch in range(epochs):
        for i in range(dataset.num_samples):
            graph, tour = dataset.graphs[i].to(device), dataset.tours[i].to(device)

            ## Forward pass and compute the loss
            output = model(graph)
            loss = criterion(output, tour)

            ## Backward pass
            optimizer.zero_grad()
            loss.backward()

            ## Update the weights
            optimizer.step()

            ## Print the epoch, sample, and loss
            print(f"Epoch {epoch + 1}, sample {i + 1}, loss: {loss.item()}")

    ##
    ## Now it's time to actually test the model with a sample from the dataset
    ##
    ## Typically, we'd use a separate test dataset, but for simplicity, we'll
    ## just use the training dataset.
    ##
    graph, tour = dataset[0]
    graph = graph.to(device)
    output = model(graph)
    print(f"Prediction: {output}\nActual: {tour}")

    ##
    ## An example of the output is below.
    ##
    ## It's actually pretty accurate for the first 4 nodes, but the last 3 are off by a bit.
    ##
    """
    Prediction: tensor([[0.9117, 0.4532],
        [0.5157, 0.7958],
        [0.8428, 0.8693],
        [0.4312, 0.8319],
        [0.5210, 0.2764]], grad_fn=<ViewBackward0>)

    Actual: tensor([[0.9200, 0.4600],
            [0.8900, 0.8800],
            [0.8400, 0.8600],
            [0.0300, 0.7400],
            [0.3800, 0.0400]])
    """

    ##
    ## Let's get an actual tour using the prediction
    ##
    predicted_real_tour = Tour.from_prediction(dataset._graphs[0].nodes, output)
    real_shortest_tour = TSPAlgorithms.get_shortest_tour(
        predicted_real_tour, TSPAlgorithm.BruteForce
    )

    ##
    ## Print the real tour and the real shortest tour
    ##
    print(f"Real tour: {predicted_real_tour}\nReal shortest tour: {real_shortest_tour}")

    ##
    ## Draw the graph with the two tours
    ##
    dataset._graphs[0].draw([predicted_real_tour, real_shortest_tour], ["pink", "red"])


##
## End of file
##
