from core.misinformation_detection import MisinformationDetection, State
from langgraph.graph import StateGraph, START, END


# Base class for the graph manager
class GraphManager:
    def __init__(self, model_name: str, system_message_prefix: str):
        self.detection_system = MisinformationDetection(
            model_name, system_message_prefix
        )
        self.graph_builder = StateGraph(State)
        self.build_graph()

    # Build the graph based on the detection system
    def build_graph(self):
        raise NotImplementedError("Subclasses should implement this method")

    # Run the graph on a single example
    def run_graph_on_example(self, example: dict):
        initial_state = {
            "messages": [],
            "article_title": example.get("article_title"),
            "article_content": example.get("article_content"),
        }
        final_state = self.graph.invoke(initial_state)
        return {
            "label": final_state.get("label"),
            "explanation": final_state.get("explanation"),
        }


# Implement the graph manager for basic classification
class BasicClassificationGraphManager(GraphManager):
    def build_graph(self):
        self.graph_builder.add_node(
            "classify_article", self.detection_system.classify_article
        )
        self.graph_builder.add_node(
            "handle_output",
            self.detection_system.handle_classification_structured_output,
        )
        self.graph_builder.add_edge(START, "classify_article")
        self.graph_builder.add_edge("classify_article", "handle_output")
        self.graph_builder.add_edge("handle_output", END)
        self.graph = self.graph_builder.compile()


class CredibilitySignalGraphManager(GraphManager):
    def build_graph(self):
        # Implement the graph for multi-prompt classification
        self.graph_builder.add_node(
            "detect_credibility_signals", self.detection_system.detect_misinformation
        )
        self.graph_builder.add_node(
            "classify_article_with_signals", self.detection_system.detect_misinformation
        )
        self.graph_builder.add_node(
            "handle_output", self.detection_system.handle_structured_output
        )
        self.graph_builder.add_edge(START, "detect_fake_news")
        self.graph_builder.add_edge("detect_fake_news", "handle_output")
        self.graph_builder.add_edge("handle_output", END)
        self.graph = self.graph_builder.compile()
