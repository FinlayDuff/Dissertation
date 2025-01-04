from core.misinformation_detection import MisinformationDetection, State
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# Base class for the graph manager
class GraphManager:
    def __init__(
        self,
        llm_model_name: str,
        article_classification_prefix: str,
        credibility_signals_prompt: dict = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.detection_system = MisinformationDetection(
            llm_model_name, article_classification_prefix, credibility_signals_prompt
        )
        self.graph_builder = StateGraph(State)
        self.build_graph()
        self.stored_credibility_signals = []

    # Build the graph based on the detection system
    def build_graph(self):
        raise NotImplementedError("Subclasses should implement this method")

    def visualize_graph(self):
        try:
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # This requires some extra dependencies and is optional
            pass

    # Run the graph on a single example
    def run_graph_on_example(self, example: dict) -> dict:
        initial_state = {
            "messages": [],
            "article_title": example.get("article_title"),
            "article_content": example.get("article_content"),
        }
        final_state = self.graph.invoke(initial_state)
        if self.verbose:
            print(final_state)

        # grab the full history and store it
        self.stored_credibility_signals.append(
            final_state.get("credibility_signals", [])
        )
        return {
            "label": final_state.get("label"),
            "confidence": final_state.get("confidence"),
            "explanation": final_state.get("explanation"),
        }


# Implement the graph manager for basic classification
class DirectClassificationGraphManager(GraphManager):
    def build_graph(self):
        self.graph_builder.add_node(
            "classify_article", self.detection_system.classify_article
        )
        self.graph_builder.add_node(
            "handle_article_classification_output",
            self.detection_system.output_article_classification,
        )
        self.graph_builder.add_edge(START, "classify_article")
        self.graph_builder.add_edge(
            "classify_article", "handle_article_classification_output"
        )
        self.graph_builder.add_edge("handle_article_classification_output", END)
        self.graph = self.graph_builder.compile()


class StagedClassificationGraphManager(GraphManager):
    def build_graph(self):
        # Implement the graph for multi-prompt classification
        self.graph_builder.add_node(
            "detect_credibility_signals",
            self.detection_system.detect_credibility_signals,
        )
        self.graph_builder.add_node(
            "classify_article_with_signals",
            self.detection_system.classify_article_with_credibility_signals,
        )
        self.graph_builder.add_node(
            "handle_credibility_signals_output",
            self.detection_system.output_credibility_signals_classification,
        )
        self.graph_builder.add_node(
            "handle_article_classification_output",
            self.detection_system.output_article_classification,
        )
        self.graph_builder.add_edge(START, "detect_credibility_signals")
        self.graph_builder.add_edge(
            "detect_credibility_signals", "handle_credibility_signals_output"
        )
        self.graph_builder.add_edge(
            "handle_credibility_signals_output", "classify_article_with_signals"
        )
        self.graph_builder.add_edge(
            "classify_article_with_signals", "handle_article_classification_output"
        )
        self.graph_builder.add_edge("handle_article_classification_output", END)
        self.graph = self.graph_builder.compile()
