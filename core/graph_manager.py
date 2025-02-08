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
        self, detection_system: MisinformationDetection, verbose: bool = False
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
        self.verbose = verbose

        # Initialize graph with state schema
        self.graph_builder = StateGraph(
            state_schema=State,  # Use our State type as schema
        )
        self.graph = None
        self.stored_credibility_signals = []
        self.build_graph()

    def build_graph(self) -> None:
        """Build the graph with clear paths to END."""
        nodes = {
            "classify_article": self.detection_system.classify_article,
            "detect_signals": self.detection_system.detect_signals,
            "make_critic_decision": self.detection_system.make_critic_decision,
            "run_followup_model": self.detection_system.run_followup_model,
        }

        # Add nodes
        for name, func in nodes.items():
            self.graph_builder.add_node(name, func)

        # Fixed paths (no conditions)
        direct_edges = [
            ("detect_signals", "classify_article"),
            ("classify_article", "make_critic_decision"),
            (
                "run_followup_model",
                "classify_article",
            ),
        ]

        for start, end in direct_edges:
            self.graph_builder.add_edge(start, end)

        # Conditional branching points
        self.graph_builder.add_conditional_edges(
            START,
            self.decide_start_path,
            {
                "classify_article": "classify_article",
                "detect_signals": "detect_signals",
            },
        )

        # Final decision point - only place that can reach END
        self.graph_builder.add_conditional_edges(
            "make_critic_decision",
            self.decide_critic_path,
            {
                "run_followup_model": "run_followup_model",
                "end": END,  # Use string "end" instead of END constant
            },
        )

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
        
    def process_credibility_signals(self, state: State) -> str:
        """
        For each credibility signal that can call a tool, call the critic to decide
        whether a follow-up tool should be executed. If the critic indicates follow-up,
        the method selects the appropriate pre-trained model to run (e.g. bias classifier,
        RAG for source analysis) and updates the state accordingly.

        Returns:
            str: The next node in the graph (e.g. "classify_article" after processing).
        """
        signals = state.get("credibility_signals", {})
        
        # Define a mapping from signal names to their follow-up tools
        followup_tools = {
            "bias": self.detection_system.bias_classifier,
            # "sources": self.detection_system.rag_tool,
            # Add additional mappings for other signals as needed.
        }
        
        for signal_name in signals:
            # Only process signals that are flagged for review.
            if signal_name in followup_tools:
                if self.verbose:
                    print(f"Evaluating credibility signal: {signal_name}")
                
                # The critic examines the current state for this signal.
                # It returns an indicator (e.g. "FOLLOW_UP" or "NO_ACTION").
                critic_decision = self.detection_system.critic(signal_name, state)
                
                if critic_decision == "FOLLOW_UP":
                    if self.verbose:
                        print(f"Critic recommends follow-up for signal: {signal_name}")
                    # Determine the appropriate tool for the signal.
                    followup_tool = followup_tools.get(signal_name)
                    if followup_tool:
                        # Run the follow-up tool; assume it updates and returns the state.
                        state = followup_tool(state)
                    else:
                        if self.verbose:
                            print(f"No follow-up tool configured for signal: {signal_name}")
                else:
                    if self.verbose:
                        print(f"Critic did not recommend follow-up for signal: {signal_name}")
        
        if self.verbose:
            print("Completed processing credibility signals; routing back to classification.")
        return "classify_article"  # Route back to classification, or change as needed.

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
                - credibility_signals: Detected signals
                - critic_decision: Result of critic analysis

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
            "few_shot": example.get("few_shot"),
            "few_shot_examples": example.get("few_shot_examples"),
        }

        final_state = self.graph.invoke(initial_state)

        if self.verbose:
            print(f"Classifcation: {final_state.get('label')}")

        self.stored_credibility_signals.append(
            final_state.get("credibility_signals", {})
        )

        return {
            "label": final_state.get("label"),
            "confidence": final_state.get("confidence"),
            "explanation": final_state.get("explanation"),
            "credibility_signals": final_state.get("credibility_signals"),
            "critic_decision": final_state.get("critic_decision"),
        }

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

    def get_stored_signals(self) -> list:
        """
        Retrieve stored credibility signals from previous runs.

        Returns:
            list: List of credibility signal dictionaries from past analyses
        """
        return self.stored_credibility_signals
