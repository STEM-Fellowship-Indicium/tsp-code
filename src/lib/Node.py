##
## Imports
##
from dataclasses import dataclass


##
## Node class
##
## If we need to add functions under the class, remove the @dataclass decorator
##
@dataclass
class Node:
    id: int
    label: str
    group: int
    title: str


##
## End of file
##
