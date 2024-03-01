##
## Imports
##
from typing import List
from node import Node
from edge import Edge
import json, datetime


##
## Graph class
##
## This class represents a graph.
##
class Graph:
    ##
    ## Constructor
    ##
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        """Initializer for the Graph class

        Args:
            nodes (List[Node]): The nodes of the graph
            edges (List[Edge]): The edges of the graph
        """
        self.nodes = nodes
        self.edges = edges

        ##
        ## End of function
        ##

    ##
    ## Convert the graph to a map
    ##
    def to_map(self) -> dict:
        """Convert the graph to a map

        Returns:
            dict: The map of the graph
        """
        return {
            "nodes": [node.to_map() for node in self.nodes],
            "edges": [edge.to_map() for edge in self.edges],
        }

        ##
        ## End of function
        ##

    ##
    ## Convert the graph to a json map
    ##
    def to_json(self) -> str:
        """Convert the graph to a json map

        Returns:
            str: The json map of the graph
        """
        return json.dumps(self.to_map())

        ##
        ## End of function
        ##

    ##
    ## Import a graph from a file
    ##
    @staticmethod
    def import_graph(filename: str) -> "Graph":
        """Import a graph from a file

        Args:
            filename (str): The name of the file to import the graph from

        Returns:
            Graph: The graph that was imported
        """
        # Open the file
        with open(filename, "r") as file:
            # Load the graph from the file
            graph = json.load(file)

        # Create the nodes and edges
        nodes = [Node(node["index"], node["x"], node["y"]) for node in graph["nodes"]]
        edges = [
            Edge(
                edge["index"],
                nodes[edge["start"]["index"]],
                nodes[edge["end"]["index"]],
            )
            for edge in graph["edges"]
        ]

        # Return the graph
        return Graph(nodes, edges)

        ##
        ## End of function
        ##

    ##
    ## Export the graph to a file
    ##
    def export(self, filename: str) -> None:
        """Export the graph to a file

        Args:
            filename (str): The name of the file to export the graph to
        """
        # Open the file (create it if it doesn't exist)
        with open(filename, "w") as file:
            json.dump(self.to_map(), file, indent=4)

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the grapg class only if we're executing THIS current file.
##
## This is so that if we import the Graph class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    # Test the graph class
    node1 = Node(0, 0, 0)
    node2 = Node(1, 1, 1)
    edge1 = Edge(0, node1, node2)
    graph = Graph([node1, node2], [edge1])
    print(json.loads(graph.to_json()))

    # todays date in the format of year-month-day-hour-minute-second
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    graph.export(f"data/graph-{date}.json")

    # Test import
    graph2 = Graph.import_graph(f"data/graph-{date}.json")
    print(json.loads(graph2.to_json()))

    # Delete the file
    import os

    os.remove(f"data/graph-{date}.json")

##
## End of file
##
