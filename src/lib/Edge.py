##
## Imports
##
from dataclasses import dataclass


##
## Edge class
##
## If we need to add functions under the class, remove the @dataclass decorator
##
@dataclass
class Edge:
    source: int
    destination: int
    weight: int


##
## End of file
##
