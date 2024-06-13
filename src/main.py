##
## Imports
##
import PySide6.QtCore
import PySide6.QtGui
from PySide6 import QtWidgets

from lib.graph import Graph
from lib.tsp.tspalgorithms import TSPAlgorithms
from lib.utils.generate_graphs import generate_graphs
from lib.utils.import_graphs import import_graphs
from lib.utils.export_graphs import export_graphs
from lib.tsp.tspvisualizations import TSPVisualizations
from lib.interfaces.tspalgorithm import TSPAlgorithm

import time


class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        ## Global variables
        self.graphs = []
        self.graph = None

        ## ## ## ## ## ## ##
        ##                ##
        ## Misc functions ##
        ##                ##
        ## ## ## ## ## ## ##

        ##
        ## 1. Function to set the response message
        ##
        def set_response_message(message: str):
            self.response_message.setText(message)

        ##
        ## 2. Function to update the number of graphs label
        ##
        def update_num_graphs_label(num_of_graphs: int):
            self.num_graphs_label.setText(f"Number of graphs: {num_of_graphs}")

        ## ## ## ## ## ## ##
        ##                ##
        ## Main functions ##
        ##                ##
        ## ## ## ## ## ## ##

        ##
        ## 1. Function to generate the graphs
        ##
        def _generate_graphs(num_graphs: int, num_nodes: int, algorithm: str):
            if num_graphs == 0:
                return set_response_message("Number of graphs cannot be 0.")

            if num_nodes == 0:
                return set_response_message("Number of nodes cannot be 0.")

            self.graphs = generate_graphs(
                num_graphs,
                num_nodes,
                algorithm,
            )

            update_num_graphs_label(len(self.graphs))

            if algorithm != TSPAlgorithm.NoneType:
                set_response_message(
                    f"Generated {num_graphs} graphs with {num_nodes} nodes each and calculated shortest path for each using {algorithm}."
                )

            else:
                set_response_message(
                    f"Generated {num_graphs} graphs with {num_nodes} nodes each."
                )

        ##
        ## 2. Function to import the graphs
        ##
        def _import_graphs(file_name: str):
            if file_name == "":
                return set_response_message(
                    "Please enter a valid file name to import the graphs from."
                )

            self.graphs = import_graphs(file_name)

            update_num_graphs_label(len(self.graphs))

            set_response_message(f"Imported graphs from {file_name}")

        ##
        ## 3. Function to export the graphs
        ##
        def _export_graphs(file_name: str):
            if file_name == "":
                return set_response_message(
                    "Please enter a valid file name to export the graphs to."
                )

            export_graphs(self.graphs, f"{file_name}")

            set_response_message(f"Exported graphs to {file_name}")

        ##
        ## 4. Function to generate a single graph
        ##
        def _generate_single_graph(num_nodes: int):
            if num_nodes == 0:
                return set_response_message("Number of nodes cannot be 0.")

            self.graph = Graph.rand(num_nodes)

            update_num_graphs_label(1)

            set_response_message(f"Generated a single graph with {num_nodes} nodes.")

        ##
        ## 5. Function to visualize the graph
        ##
        def _visualize_graph(graph: Graph, algorithm: str):
            if graph is None:
                return set_response_message(
                    "No graph to visualize. Please generate a single graph (4) before this."
                )

            if algorithm == TSPAlgorithm.BruteForce:
                TSPVisualizations.brute_force(graph)

            elif algorithm == TSPAlgorithm.GeneticAlgorithm:
                pass

            elif algorithm == TSPAlgorithm.GreedyHeuristic:
                TSPVisualizations.greedy_heuristic(graph)

            elif algorithm == TSPAlgorithm.Opt2:
                TSPVisualizations.two_opt(graph)

            elif algorithm == TSPAlgorithm.Opt3:
                TSPVisualizations.three_opt(graph)

            elif algorithm == TSPAlgorithm.SimulatedAnnealing:
                TSPVisualizations.simulated_annealing(graph)

            else:
                set_response_message("Invalid algorithm selected.")

        ##
        ## 6. Import single graph from file
        ##
        def _import_single_graph(file_name: str, graphId: str):
            if graphId == "":
                return set_response_message("Please enter a graph ID.")

            if file_name == "":
                return set_response_message(
                    "Please enter a valid file name to import the graph from."
                )

            if self.graph is not None and self.graph.id == graphId:
                return set_response_message("Graph already set to current.")

            try:
                _graph = Graph.from_cache(file_name, graphId)

                if _graph is None:
                    set_response_message(
                        f"Graph {graphId[0:50]}... not found in {file_name}"
                    )
                    return

                self.graph = _graph

                update_num_graphs_label(1)

                set_response_message(
                    f"Imported graph {graphId[0:50]}... from {file_name}"
                )

            except Exception as e:
                set_response_message(
                    f"Graph {graphId[0:50]}... not found in {file_name}"
                )

        ##
        ## 7. Export single graph to file
        ##
        def _export_single_graph(file_name: str):
            if file_name == "":
                return set_response_message(
                    "Please enter a valid file name to export the graph to."
                )

            if self.graph is None:
                return set_response_message(
                    "No graph to export. Please generate a single graph (4) before this."
                )

            self.graph.export(file_name)

            set_response_message(f"Exported graph to {file_name}")

        ##
        ## 8. Get the average speed of the algorithms
        ##
        def _get_average_speed(algorithm: str, num_graphs: int, num_nodes: int):
            if num_graphs == 0:
                return set_response_message("Number of graphs cannot be 0.")

            if num_nodes == 0:
                return set_response_message("Number of nodes cannot be 0.")

            graphs = generate_graphs(num_graphs, num_nodes)

            start = time.time()

            if algorithm == TSPAlgorithm.BruteForce:
                for graph in graphs:
                    TSPAlgorithms.brute_force(graph)

            elif algorithm == TSPAlgorithm.GeneticAlgorithm:
                for graph in graphs:
                    TSPAlgorithms.genetic_algorithm(graph)

            elif algorithm == TSPAlgorithm.GreedyHeuristic:
                for graph in graphs:
                    TSPAlgorithms.greedy_heuristic(graph)

            elif algorithm == TSPAlgorithm.Opt2:
                for graph in graphs:
                    TSPAlgorithms.two_opt(graph)

            elif algorithm == TSPAlgorithm.Opt3:
                for graph in graphs:
                    TSPAlgorithms.three_opt(graph)

            elif algorithm == TSPAlgorithm.SimulatedAnnealing:
                for graph in graphs:
                    TSPAlgorithms.simulated_annealing(graph)

            else:
                set_response_message("Invalid algorithm selected.")
                return

            end = time.time()

            set_response_message(
                f"Average speed of {algorithm} for {num_graphs} graphs with {num_nodes} nodes each: {(end - start) / num_graphs} seconds."
            )

        ## ## ## ## ## ## ##
        ##                ##
        ## The GUI layout ##
        ##                ##
        ## ## ## ## ## ## ##

        ## Layout
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        ## Text showing the number of graphs
        self.num_graphs_label = QtWidgets.QLabel("Number of graphs: 0")
        self.num_graphs_label.setFont(PySide6.QtGui.QFont("Arial", 20))
        self.layout.addWidget(self.num_graphs_label)
        self.layout.addWidget(self.num_graphs_label)

        ##
        ## 1. Generate Graphs
        ##
        self.generate_graphs_layout = QtWidgets.QHBoxLayout()
        self.generate_graphs_label = QtWidgets.QLabel("1. Generate multiple graphs")
        self.generate_graphs_layout.addWidget(self.generate_graphs_label)

        ## Number of graphs to generate
        self.generate_graphs_num_graphs_input = QtWidgets.QLineEdit()
        self.generate_graphs_num_graphs_input.setPlaceholderText("Number of graphs")
        self.generate_graphs_layout.addWidget(self.generate_graphs_num_graphs_input)

        self.generate_graphs_input_num_nodes = QtWidgets.QLineEdit()
        self.generate_graphs_input_num_nodes.setPlaceholderText("Number of nodes")
        self.generate_graphs_layout.addWidget(self.generate_graphs_input_num_nodes)

        ## Dropdown to select the algorithm to find the graph shortest tour
        self.generate_graphs_algorithm = QtWidgets.QComboBox()
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.NoneType)
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.BruteForce)
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.GeneticAlgorithm)
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.GreedyHeuristic)
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.Opt2)
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.Opt3)
        self.generate_graphs_algorithm.addItem(TSPAlgorithm.SimulatedAnnealing)
        self.generate_graphs_layout.addWidget(self.generate_graphs_algorithm)

        ## Button to generate the graphs
        self.generate_graphs_button = QtWidgets.QPushButton("Generate")
        self.generate_graphs_button.clicked.connect(
            lambda: _generate_graphs(
                int(self.generate_graphs_num_graphs_input.text() or "0"),
                int(self.generate_graphs_input_num_nodes.text() or "0"),
                self.generate_graphs_algorithm.currentText(),
            )
        )

        self.generate_graphs_layout.addWidget(self.generate_graphs_button)
        self.layout.addLayout(self.generate_graphs_layout)

        ##
        ## 2. Import Graphs
        ##
        self.import_graphs_layout = QtWidgets.QHBoxLayout()
        self.import_graphs_label = QtWidgets.QLabel("2. Import multiple graphs")
        self.import_graphs_layout.addWidget(self.import_graphs_label)

        self.import_file_dialog = QtWidgets.QFileDialog()
        self.import_file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.import_file_dialog.setNameFilter("JSON files (*.json)")

        ## File dialog open button
        self.import_graphs_open_button = QtWidgets.QPushButton(
            "Select file to import graphs from"
        )

        ## Function to open the file dialog
        def _open_file_dialog():
            if self.import_file_dialog.exec():
                file_names = self.import_file_dialog.selectedFiles()

                _import_graphs(file_names.pop())

        self.import_graphs_open_button.clicked.connect(_open_file_dialog)

        ## Add to the layout
        self.import_graphs_layout.addWidget(self.import_graphs_open_button)
        self.layout.addLayout(self.import_graphs_layout)

        ##
        ## 3. Export Graphs
        ##
        self.export_graphs_layout = QtWidgets.QHBoxLayout()
        self.export_graphs_label = QtWidgets.QLabel("3. Export multiple graphs")
        self.export_graphs_layout.addWidget(self.export_graphs_label)

        ## Input for the file name
        self.export_graphs_file_name_input = QtWidgets.QLineEdit()
        self.export_graphs_file_name_input.setPlaceholderText("File name")
        self.export_graphs_layout.addWidget(self.export_graphs_file_name_input)

        ## Button to export the graphs
        self.export_graphs_button = QtWidgets.QPushButton("Export")
        self.export_graphs_button.clicked.connect(
            lambda: _export_graphs(self.export_graphs_file_name_input.text())
        )

        ## Add to the layout
        self.export_graphs_layout.addWidget(self.export_graphs_button)
        self.layout.addLayout(self.export_graphs_layout)

        ##
        ## 4. Get the average speed of the algorithms
        ##
        self.average_speed_layout = QtWidgets.QHBoxLayout()
        self.average_speed_label = QtWidgets.QLabel(
            "4. Get the average speed of algorithms"
        )
        self.average_speed_layout.addWidget(self.average_speed_label)

        ## Input to select the number of graphs and number of nodes
        self.average_speed_num_graphs_input = QtWidgets.QLineEdit()
        self.average_speed_num_graphs_input.setPlaceholderText("Number of graphs")
        self.average_speed_layout.addWidget(self.average_speed_num_graphs_input)

        self.average_speed_num_nodes_input = QtWidgets.QLineEdit()
        self.average_speed_num_nodes_input.setPlaceholderText("Number of nodes")
        self.average_speed_layout.addWidget(self.average_speed_num_nodes_input)

        ## Dropdown to select the algorithm to find the graph shortest tour
        self.average_speed_dropdown = QtWidgets.QComboBox()
        self.average_speed_dropdown.addItem(TSPAlgorithm.BruteForce)
        self.average_speed_dropdown.addItem(TSPAlgorithm.GeneticAlgorithm)
        self.average_speed_dropdown.addItem(TSPAlgorithm.GreedyHeuristic)
        self.average_speed_dropdown.addItem(TSPAlgorithm.Opt2)
        self.average_speed_dropdown.addItem(TSPAlgorithm.Opt3)
        self.average_speed_dropdown.addItem(TSPAlgorithm.SimulatedAnnealing)
        self.average_speed_layout.addWidget(self.average_speed_dropdown)

        ## Button to get the average speed
        self.average_speed_button = QtWidgets.QPushButton("Get average speed")
        self.average_speed_button.clicked.connect(
            lambda: _get_average_speed(
                self.average_speed_dropdown.currentText(),
                int(self.average_speed_num_graphs_input.text() or "0"),
                int(self.average_speed_num_nodes_input.text() or "0"),
            )
        )

        ## Add to the layout
        self.average_speed_layout.addWidget(self.average_speed_button)
        self.layout.addLayout(self.average_speed_layout)

        ##
        ## Space between the sections
        ##
        self.layout.addWidget(QtWidgets.QLabel(""))
        self.layout.addWidget(QtWidgets.QLabel(""))

        ##
        ## 5. Generate a single graph
        ##
        self.generate_single_graph_layout = QtWidgets.QHBoxLayout()
        self.generate_single_graph_label = QtWidgets.QLabel(
            "5. Generate a single graph (sets to current)"
        )
        self.generate_single_graph_layout.addWidget(self.generate_single_graph_label)

        ## Number of nodes for the single graph
        self.generate_single_graph_num_nodes_input = QtWidgets.QLineEdit()
        self.generate_single_graph_num_nodes_input.setPlaceholderText("Number of nodes")
        self.generate_single_graph_layout.addWidget(
            self.generate_single_graph_num_nodes_input
        )

        ## Dropdown to select the algorithm to find the graph shortest tour
        self.generate_single_graph_algorithm = QtWidgets.QComboBox()
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.NoneType)
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.BruteForce)
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.GeneticAlgorithm)
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.GreedyHeuristic)
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.Opt2)
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.Opt3)
        self.generate_single_graph_algorithm.addItem(TSPAlgorithm.SimulatedAnnealing)
        self.generate_single_graph_layout.addWidget(
            self.generate_single_graph_algorithm
        )

        ## Button to generate the single graph
        self.generate_single_graph_button = QtWidgets.QPushButton("Generate")
        self.generate_single_graph_button.clicked.connect(
            lambda: _generate_single_graph(
                int(self.generate_single_graph_num_nodes_input.text() or "0")
            )
        )

        ## Add to the layout
        self.generate_single_graph_layout.addWidget(self.generate_single_graph_button)
        self.layout.addLayout(self.generate_single_graph_layout)

        ##
        ## 6. Visualize the current graph
        ##
        self.visualize_graph_layout = QtWidgets.QHBoxLayout()
        self.visualize_graph_label = QtWidgets.QLabel(
            "6. Visualize a single graph (current)"
        )
        self.visualize_graph_layout.addWidget(self.visualize_graph_label)

        ## Dropdown to select the algorithm to find the graph shortest tour
        self.visualize_graph_dropdown = QtWidgets.QComboBox()
        self.visualize_graph_dropdown.addItem(TSPAlgorithm.BruteForce)
        self.visualize_graph_dropdown.addItem(TSPAlgorithm.GeneticAlgorithm)
        self.visualize_graph_dropdown.addItem(TSPAlgorithm.GreedyHeuristic)
        self.visualize_graph_dropdown.addItem(TSPAlgorithm.Opt2)
        self.visualize_graph_dropdown.addItem(TSPAlgorithm.Opt3)
        self.visualize_graph_dropdown.addItem(TSPAlgorithm.SimulatedAnnealing)
        self.visualize_graph_layout.addWidget(self.visualize_graph_dropdown)

        ## Button to visualize the graph
        self.visualize_graph_button = QtWidgets.QPushButton("Visualize")
        self.visualize_graph_layout.addWidget(self.visualize_graph_button)
        self.visualize_graph_button.clicked.connect(
            lambda: _visualize_graph(
                self.graph, self.visualize_graph_dropdown.currentText()
            )
        )

        ## Add to the layout
        self.visualize_graph_layout.addWidget(self.visualize_graph_button)
        self.layout.addLayout(self.visualize_graph_layout)

        ##
        ## 7. Import a single graph
        ##
        self.import_single_graph_layout = QtWidgets.QHBoxLayout()
        self.import_single_graph_label = QtWidgets.QLabel(
            "7. Import a single graph (sets to current)"
        )
        self.import_single_graph_layout.addWidget(self.import_single_graph_label)

        ## Dialog to select the file
        self.import_single_graph_file_dialog = QtWidgets.QFileDialog()
        self.import_single_graph_file_dialog.setFileMode(
            QtWidgets.QFileDialog.ExistingFiles
        )
        self.import_single_graph_file_dialog.setNameFilter("JSON files (*.json)")

        ## Input for the graph ID
        self.import_single_graph_graph_id_input = QtWidgets.QLineEdit()
        self.import_single_graph_graph_id_input.setPlaceholderText("Graph ID")
        self.import_single_graph_layout.addWidget(
            self.import_single_graph_graph_id_input
        )

        ## Button to open the file dialog
        self.import_single_graph_open_button = QtWidgets.QPushButton(
            "Select file to import graph from"
        )

        ## Function to open the file dialog
        def _open_single_graph_file_dialog():
            if self.import_single_graph_file_dialog.exec():
                file_names = self.import_single_graph_file_dialog.selectedFiles()

                _import_single_graph(
                    file_names.pop(), self.import_single_graph_graph_id_input.text()
                )

        self.import_single_graph_open_button.clicked.connect(
            _open_single_graph_file_dialog
        )

        ## Add to the layout
        self.import_single_graph_layout.addWidget(self.import_single_graph_open_button)
        self.layout.addLayout(self.import_single_graph_layout)

        ##
        ## 8. Export a single graph
        ##
        self.export_single_graph_layout = QtWidgets.QHBoxLayout()
        self.export_single_graph_label = QtWidgets.QLabel(
            "8. Export a single graph (current)"
        )
        self.export_single_graph_layout.addWidget(self.export_single_graph_label)

        ## Input for the file name
        self.export_single_graph_file_name_input = QtWidgets.QLineEdit()
        self.export_single_graph_file_name_input.setPlaceholderText("File name")
        self.export_single_graph_layout.addWidget(
            self.export_single_graph_file_name_input
        )

        ## Button to export the graph
        self.export_single_graph_button = QtWidgets.QPushButton("Export")
        self.export_single_graph_button.clicked.connect(
            lambda: _export_single_graph(
                self.export_single_graph_file_name_input.text()
            )
        )

        ## Add to the layout
        self.export_single_graph_layout.addWidget(self.export_single_graph_button)
        self.layout.addLayout(self.export_single_graph_layout)

        ## Response message text
        self.response_message = QtWidgets.QLabel("")
        self.response_message.setFont(PySide6.QtGui.QFont("Arial", 14))
        self.layout.addWidget(self.response_message)


##
## Run the main application
##
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication([])

    widget = MainWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())


##
## End of file
##
