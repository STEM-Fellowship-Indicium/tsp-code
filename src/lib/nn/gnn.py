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

        self.fc1 = nn.Linear(node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

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


        Example Output:
            tensor([[15., 43.], [20., 17.], [47., 13.], [94., 49.], [49., 81.]])
        """
        ##
        ## We want the output to be positions of the nodes. We'll then use these
        ## distances to locate the nearest node to each node.
        ##
        ## The output shape should be as shown in the example above: (num_nodes, 2)
        ##

        ##
        ## Perform the forward pass.
        ##
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        ##
        ## Reshape the output. We want the output to be of shape (num_nodes, 2).
        ##
        x = x.view(-1, self.output_dim)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the graph neural network dataset
    from graphdataset import GraphDataset

    dataset = GraphDataset(num_samples=10, num_nodes=5, node_features=2)

    # Create the graph neural network model
    model = GNN(node_features=2, hidden_dim=16, output_dim=2).to(device)

    # Create a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(1000):
        for i in range(dataset.num_samples):
            graph, tour = dataset.graphs[i].to(device), dataset.tours[i].to(device)

            # Forward pass
            output = model(graph)

            # Compute the loss
            loss = criterion(output, tour)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update the weights
            optimizer.step()

            print(f"Epoch {epoch + 1}, sample {i + 1}, loss: {loss.item()}")

    # Make a prediction
    graph, tour = dataset[0]
    graph = graph.to(device)
    output = model(graph)
    print(f"Prediction: {output}\nActual: {tour}")


##
## End of file
##
