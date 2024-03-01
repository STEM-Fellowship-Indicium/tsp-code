##
## Imports
##
from dataclasses import dataclass


##
## Graph class
##
## If we need to add functions under the class, remove the @dataclass decorator
##
@dataclass
class Graph:
    nodes: list
    edges: list


##
## End of file
##
