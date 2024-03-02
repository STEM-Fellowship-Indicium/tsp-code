##
## Adjust to relative path
##
if __name__ == "__main__":
    import sys

    sys.path.append("src")


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
    def __init__(self, index: int, x: int, y: int) -> None:
        """Initializer for the Node class

        Args:
            index (int): The index of the node
            x (int): The x coordinate of the node
            y (int): The y coordinate of the node
        """
        self.index = index
        self.x = x
        self.y = y

        ##
        ## End of function
        ##

    ##
    ## String representation of the node
    ##
    def __str__(self) -> str:
        """String representation of the node

        Returns:
            _type_: The string representation of the node
        """
        return f"({self.x}, {self.y})"

        ##
        ## End of function
        ##

    ##
    ## Convert the node to a map
    ##
    def to_map(self) -> dict:
        """Convert the node to a map

        Returns:
            dict: The map of the node
        """
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
        """Convert the node to a json map

        Returns:
            str: The json map of the node
        """
        return json.dumps(self.to_map())

        ##
        ## End of function
        ##

    ##
    ## Normalize the node
    ##
    ## This takes a node and normalizes its x and y values to be between 0 and 1
    ##
    def normalize(self) -> None:
        """Normalize the node"""
        min_x = min_y = 0
        max_x = max_y = 100

        self.x = (self.x - min_x) / (max_x - min_x)
        self.y = (self.y - min_y) / (max_y - min_y)

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
