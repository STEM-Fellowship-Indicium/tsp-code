##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")

##
## Imports
##
## pip install torch-geometric
##
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import torch_scatter

##
## GNN class
##
class GeometricGNN(MessagePassing):
    ##
    ## Constructor
    ##
    def __init__(self, node_dim: int, hidden_dim: int, num_nodes: int) -> None:
        """Initializer for the GeometricGNN class

        Args:
            node_dim (int): The number of features for each node
            hidden_dim (int): The hidden dimension for the GeometricGNN
        """
        super(GeometricGNN, self).__init__(aggr="mean")  ## Mean aggregation

        ##
        ## Variables
        ##
        self._node_dim = node_dim
        self._hidden_dim = hidden_dim
        self._num_nodes = num_nodes

        ##
        ## Create the sequence of layers
        ##
        self.input_layer = torch.nn.Linear(node_dim, hidden_dim)
        self.hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, node_dim)

        ##
        ## End of function
        ##

    ##
    ## Message Passing
    ##
    def message(self, x_j):
        """Function to compute messages"""
        return x_j

    ##
    ## Aggregate Messages
    ##
    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter_mean(inputs, index, dim=0)

    ##
    ## Forward pass
    ##
    def forward(self, batch_data: Batch) -> torch.Tensor:
        """Forward pass for the GeometricGNN

        Args:
            batch_data (Batch): The batch data

        Returns:
            torch.Tensor: The output tensor
        """
        ## Get data
        x, edge_index = batch_data.x, batch_data.edge_index

        ## Apply input layer
        x = F.relu(self.input_layer(x))

        ## Message passing
        x = self.propagate(edge_index, x=x)

        ## Apply hidden layer
        x = F.relu(self.hidden_layer(x))

        ## Apply output layer
        x = self.output_layer(x)

        ## Return the output
        return x

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## Testing. This tests only the GeometricGNN class if we were to run the file.
## This prevents the test from running if we were to import the file.
##
if __name__ == "__main__":
    ## Create the model and optimizer
    model = GeometricGNN(node_dim=2, hidden_dim=16, num_nodes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    ##
    ## Graph 1
    ##
    edge_index_1 = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    ).t().contiguous()  ## Edges (node indexes)

    nodes_1 = torch.tensor(
        [[0, 0], [1, 0], [2, 0]], dtype=torch.float
    )  ## Node positions (x, y)

    tour_1 = torch.tensor(
        [[0, 0], [1, 0], [2, 0]], dtype=torch.float
    )  ## Tour of the nodes (output)

    graph_1 = Data(x=nodes_1, y=tour_1, edge_index=edge_index_1)

    ##
    ## Graph 2
    ##
    edge_index_2 = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    ).t().contiguous()  ## Edges (node indexes)
    nodes_2 = torch.tensor(
        [[0, 0], [1, 0], [2, 0]], dtype=torch.float
    )  ## Node positions (x, y)
    tour_2 = torch.tensor(
        [[0, 0], [1, 0], [2, 0]], dtype=torch.float
    )  ## Tour of the nodes (output)
    graph_2 = Data(x=nodes_2, y=tour_2, edge_index=edge_index_2)

    ## Batch the graphs
    batch = Batch.from_data_list([graph_1, graph_2])
    epochs = 100

    ##
    ## Train the model
    ##
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(batch)

        ## We need to use a custom loss function for the TSP
        ## compute_tour_length(output, data)  ## Custom loss function
        loss = loss_fn(out, batch.y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    ##
    ## Test the model
    ##
    edge_index_3 = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    ).t().contiguous()  ## Edges (node indexes)

    nodes_3 = torch.tensor(
        [[0, 0], [1, 0], [2, 0]], dtype=torch.float
    )  ## Node positions (x, y)

    tour_3 = torch.tensor(
        [[0, 0], [1, 0], [2, 0]], dtype=torch.float
    )  ## Tour of the nodes (output)

    graph_3 = Data(x=nodes_3, y=tour_3, edge_index=edge_index_3)

    out = model(graph_3)
    print(out.detach())

    """
    Example output:
    
    tensor([[ 1.2296e-03, -1.7348e-03],
        [ 9.9695e-01, -4.6250e-03],
        [ 1.9982e+00,  5.5105e-03]])

    = (approximately)

    [[0, 0],
    [1, 0],
    [2, 0]]

    Which when you compare to the tour_3 tensor, you can see that the model has learned the tour.
    """

##
## End of file
##
