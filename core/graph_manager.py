"""
Graph Manager Module

This module provides graph-based workflow management for misinformation detection.
It constructs and manages a directed graph of operations that:
- Classifies articles
- Detects credibility signals
- Makes routing decisions based on state
- Handles conditional branching between operations
"""

from langgraph.graph import StateGraph, END, START
from IPython.display import Image, display

from core.misinformation_detection import MisinformationDetection
from core.state import State
from typing import Dict


class GraphManager:
    """
    Manages the workflow graph for misinformation detection operations.

    This class constructs and executes a directed graph of operations that can:
    - Route between different classification methods
    - Handle credibility signal detection
    - Make decisions about additional analysis
    - Store results for later analysis

    The graph uses a State object to maintain the workflow context and results.
    """

    def __init__(
        self,
        detection_system: MisinformationDetection,
        experiment_config: Dict,
        verbose: bool = False,
    ):
        """
        Initialize the graph manager.

        Args:
            detection_system: MisinformationDetection instance to handle operations
            verbose: Whether to print debug information during execution

        Example:
            >>> detector = MisinformationDetection()
            >>> manager = GraphManager(detector, verbose=True)
        """
        self.detection_system = detection_system
        self._experiment_config = experiment_config
        self.verbose = verbose

        # Initialize graph with state schema
        self.graph_builder = StateGraph(
            state_schema=State,  # Use our State type as schema
        )
        self.graph = None
        self.stored_states = []
        self.build_graph()

    def _get_graph_configuration(self):
        required_nodes = ["classify_article"]
        direct_edges = [("classify_article", END)]
        conditional_edges = []

        signals_config = self._experiment_config.get("signals", {})
        if signals_config.get("enabled", False):
            required_nodes.append("detect_signals")
            direct_edges.append((START, "detect_signals"))
            if signals_config.get("critic", False):
                required_nodes.append("classify_topic")
                direct_edges.append(("detect_signals", "classify_topic"))
                required_nodes.append("critic_credibility_signals")
                direct_edges.append(("classify_topic", "critic_credibility_signals"))
                if signals_config.get("followup", False):
                    required_nodes.append("run_followup_analysis")
                    direct_edges.append(
                        (
                            "run_followup_analysis",
                            "classify_article",
                        )
                    )
                    conditional_edges.append(
                        (
                            "critic_credibility_signals",
                            self.decide_signals_critic_path,
                            {
                                "classify_article": "classify_article",
                                "run_followup_analysis": "run_followup_analysis",
                            },
                        )
                    )
                else:
                    direct_edges.append(
                        ("critic_credibility_signals", "classify_article")
                    )
            else:
                direct_edges.append(("detect_signals", "classify_article"))

        else:
            direct_edges.append((START, "classify_article"))
        return required_nodes, direct_edges, conditional_edges

    def build_graph(self) -> None:
        """Build the graph with clear paths to END."""
        # All possible nodes
        nodes = {
            "classify_article": self.detection_system.classify_article,
            "detect_signals": self.detection_system.detect_signals,
            "critic_credibility_signals": self.detection_system.critic_signal_classification,
            "run_followup_analysis": self.detection_system.run_followup_analysis,
            "classify_topic": self.detection_system.classify_topic,
        }

        required_nodes, direct_edges, conditional_edges = (
            self._get_graph_configuration()
        )
        # Add nodes to graph
        for node_name in required_nodes:
            self.graph_builder.add_node(node_name, nodes[node_name])

        # Add direct edges between nodes
        for start, end in direct_edges:
            self.graph_builder.add_edge(start, end)

        # Add conditional edges between nodes
        for start, condition, edges in conditional_edges:
            self.graph_builder.add_conditional_edges(start, condition, edges)

        self.graph = self.graph_builder.compile()

    def decide_start_path(self, state: State) -> str:
        """Determine initial path based on whether to use signals."""
        try:
            if state.get("use_signals"):
                if self.verbose:
                    print("Starting with signal detection")
                return "detect_signals"

            if self.verbose:
                print("Starting with direct classification")
            return "classify_article"

        except Exception as e:
            print(f"Error deciding start path: {e}")
            return "classify_article"  # Default to direct classification on error

    def decide_critic_path(self, state: State) -> str:
        """Determine path after critic decision."""
        try:
            if state.get("critic_decision") == "FAILED":
                if self.verbose:
                    print("Critic decided to run followup model")
                return "run_followup_model"

            if self.verbose:
                print("Critic decided to end analysis")
            return "end"  # Use string "end" instead of END constant

        except Exception as e:
            print(f"Error in critic decision: {e}")
            return "end"  # Default to ending on error

    def decide_signals_critic_path(self, state: State) -> str:
        """
        Args:
            state: Current state with article information

        Returns:
            str: The next node in the graph (e.g. "classify_article" after processing).
        """
        signals_critiques = state.get("signals_critiques")
        if signals_critiques.get("followup_signals", False):
            return "run_followup_analysis"

        return "classify_article"

    def run_graph_on_example(self, example: dict) -> dict:
        """
        Run the graph workflow on a single example.

        Args:
            example: Dictionary containing:
                - article_title: Title of article
                - article_content: Content to analyze
                - experiment_name: Name of experiment config
                - use_signals: Whether to use credibility signals
                - use_bulk_signals: Whether to detect signals in bulk
                - few_shot: Whether to use few-shot examples
                - few_shot_examples: List of examples if few_shot is True

        Returns:
            dict: Results containing:
                - label: Classification result
                - confidence: Confidence score
                - explanation: Reasoning

        Example:
            >>> example = {
            ...     "article_title": "Example News",
            ...     "article_content": "Article text...",
            ...     "experiment_name": "baseline_gpt4"
            ... }
            >>> results = manager.run_graph_on_example(example)
        """

        initial_state = {
            "messages": [],
            "article_title": example.get("article_title"),
            "article_content": example.get("article_content"),
            "experiment_name": example.get("experiment_name"),
            "use_signals": example.get("use_signals"),
            "use_bulk_signals": example.get("use_bulk_signals"),
            "condensed_signals": example.get("condensed_signals"),
            "few_shot": example.get("few_shot"),
            "few_shot_examples": example.get("few_shot_examples"),
        }

        final_state = self.graph.invoke(initial_state)

        if self.verbose:
            print(f"Classification: {final_state.get('label')}")

        save_final_state = {
            k: v for k, v in final_state.items() if k not in initial_state.keys()
        }

        return save_final_state

    def visualize_graph(self) -> None:
        """
        Display the graph visualization in notebook environments.

        Generates a Mermaid diagram showing the nodes and edges of the workflow graph.
        Only works in Jupyter notebook environments.
        """
        try:
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            print("Graph visualization only available in notebook environment")
