##
## path: src/lib/example.py
##

##
## Imports
##
from torch import Tensor
from torch.nn import (
    Module as NNModule,
    Linear,
    Conv2d,
    MaxPool2d,
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
    ## The number of input channels and the number of classes
    ## are used to define the input and output size of the network
    ##
    ## The input size is the number of nodes in the graph
    ## The output size is the number of nodes in the graph
    ##
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(CNN, self).__init__()

        self.sequence = Sequential(
            ##
            ## Convolutional layer 1
            ##
            ## We use the provided in channels with 8 output channels
            ## The kernel size is 3x3 with a stride of 1 and padding of 1 to
            ## maintain the input shape
            ##
            Conv2d(in_channels, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ##
            ## Activation function
            ##
            ReLU(),
            ##
            ## Max pooling layer
            ##
            ## We use a kernel size of 2x2 and a stride of 2 to reduce the
            ## input size by half
            ##
            ## This is done to reduce the number of parameters and computation
            ## in the network. (faster training and less overfitting)
            ##
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ##
            ## Convolutional layer 2
            ##
            Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ##
            ## Activation function
            ##
            ReLU(),
            ##
            ## Max pooling layer
            ##
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            ##
            ## Flatten the input tensor
            ##
            ## This is done to convert the 2D tensor into a 1D tensor
            ## to be used as input for the fully connected layer
            ##
            Flatten(),
            ##
            ## Fully connected layer
            ##
            ## The input size is 16 * 7 * 7 because of the max pooling layers
            ## The output size is the number of classes
            ##
            Linear(16 * 7 * 7, num_classes),
        )

    ##
    ## End of init function
    ##

    def forward(self, x: Tensor) -> Tensor:
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
