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
from typing import List
from lib.node import Node
from lib.edge import Edge
from lib.tour import Tour
from lib.utils import generate_nodes


##
## Graph class
##
## This class represents a graph.
##
class Graph:
    ##
    ## Constructor
    ##
    def __init__(
        self,
        edges: List[Edge],
        nodes: List[Node] = [],
        adj_matrix: List[List[int]] = [],
        shortest_tour: Tour = None,
    ) -> None:
        """Initializer for the Graph class

        Args:
            nodes (List[Node]): The nodes of the graph
            edges (List[Edge]): The edges of the graph
            adj_matrix (List[List[int]]): The adjacency matrix of the graph
            shortest_tour (Tour): The shortest path of the graph
        """

        self.edges = edges
        self.shortest_tour = shortest_tour
        self.adj_matrix = adj_matrix

        self.nodes = nodes
        if len(self.nodes) == 0:
            self.set_nodes_to_edges()

        self.node_idxs = [node.idx for node in nodes]
        self.set_adj_matrix()

        ##
        ## End of function
        ##

    ##
    ## String representation of the graph
    ##
    def __str__(self) -> str:
        """String representation of the graph

        Returns:
            _type_: The string representation of the graph
        """
        return f"{[str(node) for node in self.nodes]}, {[str(edge) for edge in self.edges]}"

        ##
        ## End of function
        ##

    ##
    ## Generate a random graph
    ##
    ## We'll use our custom generate_nodes function to generate a random
    ## graph with n nodes.
    ##
    @staticmethod
    def rand(num_nodes: int) -> "Graph":
        """Generate a random graph

        Args:
            num_nodes (int): The number of nodes to generate

        Returns:
            Graph: The random graph
        """

        # Generate the nodes
        nodes = generate_nodes(num_nodes)

        # Create the edges
        edges = [
            Edge(idx, nodes[i], nodes[j])
            for idx, i in enumerate(range(num_nodes))
            for j in range(i + 1, num_nodes)
        ]

        # Return the graph
        return Graph(edges, nodes)

        ##
        ## End of function
        ##

    ##
    ## Import a graph from a file
    ##
    @staticmethod
    def import_json(filename: str) -> "Graph":
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
        adj_matrix = graph["adj_matrix"]
        shortest_tour = Tour.from_map(graph["shortest_tour"])

        nodes = [Node(node["idx"], node["x"], node["y"]) for node in graph["nodes"]]
        edges = [
            Edge(
                edge["idx"],
                nodes[edge["start"]["idx"]],
                nodes[edge["end"]["idx"]],
                edge["weight"],
            )
            for edge in graph["edges"]
        ]

        # Return the graph
        return Graph(edges, nodes, adj_matrix, shortest_tour)

        ##
        ## End of function
        ##

    ##
    ## Set the nodes to the edges (start and end nodes of the edges)
    ##
    def set_nodes_to_edges(self, edges: List[Node] = []) -> List[Node]:
        """Set the nodes to the edges (start and end nodes of the edges)

        Args:
            edges (List[Node], optional): The edges to set the nodes to. Defaults to [] which uses current edges.

        Returns:
            List[Node]: The previous nodes
        """

        # Save the previous nodes
        self.prev_nodes = self.nodes
        self.nodes = []

        # Use the given edges if they exist
        if len(edges) > 0:
            self.edges = edges

        # Add the start and end nodes of the edges to the nodes list
        for edge in self.edges:
            if edge.start not in self.nodes:
                self.nodes.append(edge.start)
            if edge.end not in self.nodes:
                self.nodes.append(edge.end)

        # Return the previous nodes
        return self.prev_nodes

        ##
        ## End of function
        ##

    ##
    ## Create adjacency matrix
    ##
    def set_adj_matrix(self) -> None:
        """Create adjacency matrix"""
        num_nodes = len(self.nodes)

        # Fill the adj matrix with a bunch of arrays of 0s
        self.adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

        # Fill the adj matrix with the weights of the edges
        for edge in self.edges:
            start = edge.start.idx
            end = edge.end.idx

            # Set the weight of the edge in the adj matrix
            self.adj_matrix[start][end] = edge.weight
            self.adj_matrix[end][start] = edge.weight

        ##
        ## End of function
        ##

    ##
    ## Get the node with the given idx
    ##
    def get_node(self, idx: int) -> Node:
        """Get the node with the given idx

        Args:
            idx (int): The idx of the node to get

        Returns:
            Node: The node with the given idx
        """
        return next(node for node in self.nodes if node.idx == idx)

        ##
        ## End of function
        ##

    ##
    ## Get the edge with the given idx
    ##
    def get_edge(self, idx: int) -> Edge:
        """Get the edge with the given idx

        Args:
            idx (int): The idx of the edge to get

        Returns:
            Edge: The edge with the given idx
        """
        return next(edge for edge in self.edges if edge.idx == idx)

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
            "adj_matrix": self.adj_matrix,
            "shortest_tour": (
                self.shortest_tour.to_map() if self.shortest_tour else None
            ),
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
    ## Print the graph
    ##
    def print(self) -> None:
        """Print the graph"""
        print(f"Nodes: {[str(node) for node in self.nodes]}")
        print(f"Edges: {[str(edge) for edge in self.edges]}")
        print(f"Shortest tour: {self.shortest_tour}")
        print(f"Adjacency matrix: {self.adj_matrix}")

        ##
        ## End of function
        ##

    ##
    ## End of class
    ##


##
## This tests the graph class only if we're executing THIS current file.
##
## This is so that if we import the Graph class from another file, this
## code (in the 'if' statement) won't run.
##
if __name__ == "__main__":
    import datetime

    # Test the graph class
    node1 = Node(0, 0, 0)
    node2 = Node(1, 1, 1)
    edge1 = Edge(0, node1, node2)
    graph1 = Graph([edge1], [node1, node2])
    graph1.print()

    # todays date in the format of year-month-day-hour-minute-second
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    graph1.export(f"data/graph-{date}.json")

    # Test import
    graph2 = Graph.import_json(f"data/graph-{date}.json")
    graph2.print()

    # Delete the file
    import os

    # os.remove(f"data/graph-{date}.json")


##
## End of file
##
