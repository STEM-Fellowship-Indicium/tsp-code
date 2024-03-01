##
## Imports
##
import json


##
## Node class
##
## This class represents a node in a graph.
##
class Node:
    ##
    ## Constructor
    ##
    def __init__(self, index: int, x: int, y: int):
        self.index = index
        self.x = x
        self.y = y

        ##
        ## End of function
        ##

    ##
    ## String representation of the node
    ##
    def __str__(self):
        return f"({self.x}, {self.y})"

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a map
    ##
    def to_map(self) -> dict:
        return {
            "index": self.index,
            "x": self.x,
            "y": self.y,
        }

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a json map
    ##
    def to_json(self) -> str:
        return json.dumps(self.to_map())

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the node class only if we're executing THIS current file.
##
## This is so that if we import the Node class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    node = Node(0, 1, 2)
    print(node)
    print(node.to_json())

##
## End of file
##
