##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


##
## Imports
##
from torch import Tensor
from torch.nn import (
    Module as NNModule,
    Linear,
    Conv2d,
    Flatten,
    Sequential,
    ReLU,
)


##
## Pytorch Convolutional Neural Network
##
## This will be used to take in a matrix of nodes and output a prediction
## for the shortest tour.
##
class CNN(NNModule):
    ##
    ## The number of input channels are used to define the input size
    ## and the number of output channels are used to define the output size.
    ##
    ## The input size is the number of nodes in the graph
    ## The output size is the number of nodes in the graph
    ##
    def __init__(self, in_channels: int) -> None:
        """Initializer for the CNN class

        Args:
            in_channels (int): The number of input channels

        Raises:
            ValueError: If the number of input channels is less than 1
        """
        super(CNN, self).__init__()

        if in_channels < 1:
            raise ValueError("Invalid number of input channels")

        ##
        ## Define the sequence that will be used to take in a graph
        ## and return the shortest tour of nodes.
        ##
        ## This is done by using a test dataset of graphs and their
        ## shortest tours to train the network to predict the shortest
        ## tour of a graph.
        ##
        ## The sequence is defined as follows:
        ##
        self.sequence = Sequential(
            ##
            ## Convolutional layer 1
            ##
            ## We use the provided in channels with 32 output channels
            ## The kernel size is 3x3 with a stride of 1 and padding of 1 to
            ## maintain the input shape
            ##
            Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            ##
            ## Since we're already using a 1 channel input, we don't need
            ## a max pooling layer. We've already reduced the input size
            ## to the lowest possible which optimizes the network.
            ##
            ## There's also no need for another convolutional layer.
            ##
            ## We'll instead just return the output
            ##
            Flatten(),
            Linear(20, 32),
        )

    ##
    ## End of init function
    ##

    ##
    ## The forward function is used to define the forward pass of the network
    ##
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the network

        Args:
            x (Tensor): The input tensor

        Returns:
            Tensor: The output tensor
        """
        return self.sequence(x)

        ##
        ## End of forward function
        ##

    ##
    ## End of class
    ##


##
## End of file
##

##
## Execute the test
##
if __name__ == "__main__":
    from lib.utils import generate_points

    # Create a new CNN with 1 input channel
    cnn = CNN(1)

    # Generate points then convert the input to 3D tensor with 1 channel
    input = Tensor(generate_points(10))
    input = input.view(1, 2, 10)

    # Pass the input through the network
    output = cnn.forward(input)
    print(output)

    # Set the correct answer for the test
    # correct = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Print the output
    print(output)
