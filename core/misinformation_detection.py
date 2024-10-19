from typing_extensions import Annotated, TypedDict
from utils.langchain.llm_model_selector import get_llm_from_model_name
from core.prompts import STRUCTURED_OUTPUT_PROMPT


# Define the state
class State(TypedDict):
    messages: Annotated[list, "A list to store messages exchanged with the LLM"]
    article_title: Annotated[str, "The title of the article to be analysed"]
    article_content: Annotated[str, "The content of the article to be analysed"]
    label: Annotated[str, "The label of the classification ('Credible' or 'Fake')"]
    explanation: Annotated[str, "The reasoning behind the classification"]


# misinformation_detection.py


class MisinformationDetectionBase:
    def __init__(self, model_name: str):
        self.llm = get_llm_from_model_name(model_name)

    def classify_article(self, article_content):
        raise NotImplementedError("Subclasses should implement this method")

    def handle_classification_structured_output(self, state: State):
        raw_output = state["messages"][-1].content
        label_start = raw_output.find("Label: ") + len("Label: ")
        label_end = raw_output.find("\n", label_start)
        label = raw_output[label_start:label_end].strip()

        explanation_start = raw_output.find("Explanation:") + len("Explanation:")
        explanation = raw_output[explanation_start:].strip()

        state["label"] = label
        state["explanation"] = (
            explanation if explanation else "No explanation provided."
        )
        return state


# Implement the misinformation detection system
class MisinformationDetection:
    def __init__(self, model_name: str, system_message_prefix: str):
        self.llm = get_llm_from_model_name(model_name)
        self.system_message = f"""
            {system_message_prefix}\n
            {STRUCTURED_OUTPUT_PROMPT}
        """

    # Method to detect misinformation in an article
    def classify_article(self, state: State):
        """
        For a given article, detect whether it is credible or fake using the LLM.
        The LLM's system message sets up how it should break down the problem and strutctured output.
        """
        article_title = state["article_title"]
        article_content = state["article_content"]
        input_text = f"Title: {article_title}\n\nContent: {article_content}"

        response = self.llm.invoke(
            [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input_text},
            ]
        )

        state["messages"] = [response]  # Update state with response
        return state

    # Method to handle the structured output from the LLM
    def handle_classification_structured_output(self, state: State):
        raw_output = state["messages"][-1].content
        label_start = raw_output.find("Label: ") + len("Label: ")
        label_end = raw_output.find("\n", label_start)
        label = raw_output[label_start:label_end].strip()

        explanation_start = raw_output.find("Explanation:") + len("Explanation:")
        explanation = raw_output[explanation_start:].strip()

        state["label"] = label
        state["explanation"] = (
            explanation if explanation else "No explanation provided."
        )
        return state
