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
## Linear GNN class
##
class LinearGNN(nn.Module):
    ##
    ## Constructor
    ##
    def __init__(
        self, input_features: int = 2, hidden_dim: int = 16, output_features: int = 2
    ) -> None:
        """Initializer for the LinearGNN class

        Args:
            input_features (int): The number of features for each node
            hidden_dim (int): The hidden dimension for the LinearGNN
            output_features (int): The output dimension for the LinearGNN
        """
        super(LinearGNN, self).__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_features = output_features

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
            nn.Linear(input_features, hidden_dim),
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
            nn.Linear(hidden_dim, output_features),
        )

        ##
        ## End of function
        ##

    ##
    ## Forward pass
    ##
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the LinearGNN

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The output of the LinearGNN

        """
        ##
        ## We want the output to be positions of the nodes. We'll then use these
        ## distances to locate the nearest node to each node.
        ##
        ## We'll take the output of the LinearGNN and each row will be the x and y
        ## coordinates of a node.
        ##
        ## Perform the forward pass.
        ##
        x = self.sequence(x)

        ##
        ## Reshape the output. We want the output to be of shape (num_nodes + 1, self.output_features) [(rows, columns)]
        ##
        ## Note that: -1 automatically calculates the number of rows needed.
        ##
        # x = x.view(-1, self.output_features)

        ##
        ## Return the output
        ##
        return x

    ##
    ## End of class
    ##


##
## This tests the LinearGNN class only if we're executing THIS current file.
##
## This is so that if we import the LinearGNN class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    ##
    ## Test Imports
    ##
    from lib.nn.graphdataset import GraphDataset
    from lib.tour import Tour
    from lib.tsp.tspalgorithms import TSPAlgorithms, TSPAlgorithm

    ##
    ## If our device has a GPU, use it (for speed).
    ##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##
    ## Create a dataset using the custom GraphDataset class
    ##
    dataset = GraphDataset(num_samples=1000, num_nodes=7, node_features=2)

    ##
    ## Train the model
    ##
    def train() -> None:
        ##
        ## Create the model
        ##
        model = LinearGNN(input_features=2, hidden_dim=16, output_features=2).to(device)

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
        ## Train the model with 100 epochs
        ##
        epochs: int = 100
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
        ## Save the model
        ##
        # torch.save(model.state_dict(), "data/nn/gnn/linear.pth")

    ##
    ## Train the model
    ##
    # train()

    ##
    ## Now it's time to actually test the model with a sample from the dataset
    ##
    ## Typically, we'd use a separate test dataset, but for simplicity, we'll
    ## just use the training dataset.
    ##
    model = LinearGNN(input_features=2, hidden_dim=16, output_features=2).to(device)
    model.load_state_dict(torch.load("data/nn/gnn/linear.pth"))
    model.eval()

    ##
    ## Get a sample from the dataset
    ##
    graph, tour = dataset[0]
    graph = graph.to(device)
    tour = tour.to(device)

    ##
    ## Make a prediction
    ##
    output = model(graph)

    ##
    ## Let's get an actual tour using the prediction
    ##
    graph = dataset._graphs[0]

    pred_tour = Tour.from_prediction(graph.nodes, output)

    actual_tour = TSPAlgorithms.get_shortest_tour(graph, TSPAlgorithm.BruteForce)

    ##
    ## Print the predicted tour and the actual shortest tour
    ##
    print(f"Predicted tour: {pred_tour}\nActual tour: {actual_tour}")

    ##
    ## Draw the graph with the two tours
    ##
    graph.draw([pred_tour, actual_tour], ["pink", "red"])  ##


##
## End of file
##
