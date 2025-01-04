from typing_extensions import Annotated, TypedDict
from utils.langchain.llm_model_selector import get_llm_from_model_name
from core.prompts import (
    STRUCTURED_OUTPUT_PROMPT_ARTICLE,
    STRUCTURED_OUTPUT_PROMPT_SIGNAL,
    CREDIBILITY_SIGNALS_CLASSIFCIATION,
)
import json


# Define the state
class State(TypedDict):
    messages: Annotated[list, "A list to store messages exchanged with the LLM"]
    article_title: Annotated[str, "The title of the article to be analysed"]
    article_content: Annotated[str, "The content of the article to be analysed"]
    credibility_signals: Annotated[
        dict[str, dict],
        "A dictionary of credibility signals with metadata such as confidence, classification, and reasoning",
    ]
    label: Annotated[str, "The label of the classification ('Credible' or 'Fake')"]
    confidence: Annotated[float, "The confidence score of the classification"]
    explanation: Annotated[str, "The reasoning behind the classification"]


# Implement the misinformation detection system
class MisinformationDetection:
    def __init__(
        self,
        llm_model_name: str,
        article_classification_prefix: str,
        credibility_signals_prompt: dict = None,
    ):
        self._llm = get_llm_from_model_name(llm_model_name)
        self._article_classification_prefix = article_classification_prefix
        # We initialize the system message for article classification. This does not change in an experiment.
        if credibility_signals_prompt:
            self._credibility_signals_prompt = credibility_signals_prompt
            self._initialise_credibility_signals_system_message()

    def _generate_article_classification_prompt(
        self, article_title: str, article_content: str
    ) -> list[dict]:

        system_message = f"""
            {self._article_classification_prefix}\n
            {STRUCTURED_OUTPUT_PROMPT_ARTICLE}
        """

        input_text = f"Title: {article_title}\n\nContent: {article_content}"
        chat_input = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
        ]

        return chat_input

    def _generate_article_classification_prompt_with_credibility_signals(
        self, article_title: str, article_content: str, credibility_signals: dict
    ) -> list[dict]:
        credibility_signals_str = json.dumps(credibility_signals, indent=4)
        article_classification_credibility_signals = f"""
            You are an expert fact-checker tasked with determining whether an article should be classified as “fake” (0) or “real” (1). 
            Your decision must be based on the article’s content and an evaluation of credibility signals, which include their classification, explanations, and confidence levels. 
            Follow these steps carefully:
            1.	Analyse the Article Content:
                - Read the content of the article thoroughly.
                - Look for consistency, evidence, and logical coherence in the claims.
            2.	Incorporate Credibility Signals:
                - Review the credibility signal classifications provided. These include:
                    - Signal Name: [e.g., Evidence, Bias, Source Credibility, Emotional Valence, etc.]
                    - Classification: [e.g., Present/Absent or Positive/Negative/Neutral]
                    - Explanation: Detailed reasoning for the classification of each signal.
                    - Confidence: A float indicating the confidence level of the signal classification.
            3.	Synthesise Information:
                - Combine your analysis of the article's content with the credibility signals to assess the overall credibility of the article.
            4.	Provide a Final Classification:
                - Classify the article as “fake” (0) or “real” (1) based on the evidence and your reasoning.
                - Justify your decision with a clear and concise explanation that references both the article content and the credibility signals.
                - Provide a confidence score between 0 and 1 to indicate the certainty of your classification.
            {credibility_signals_str}
        """

        system_message = f"""
            {article_classification_credibility_signals}\n
            {STRUCTURED_OUTPUT_PROMPT_ARTICLE}
        """

        input_text = f"Title: {article_title}\n\nContent: {article_content}"
        chat_input = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
        ]

        return chat_input

    def _generate_credibility_signals_batch(self):
        batch_prompt = CREDIBILITY_SIGNALS_CLASSIFCIATION
        for signal_key, signal_prompt in self._credibility_signals_prompt.items():
            batch_prompt += f"{signal_key}: {signal_prompt}\n\n"

        return batch_prompt

    def _initialise_credibility_signals_system_message(self):

        prompt = self._generate_credibility_signals_batch()
        system_message = f"""
            {prompt}\n
            {STRUCTURED_OUTPUT_PROMPT_SIGNAL}
        """
        self._credibility_signals_classification_system_message = system_message

    def _generate_credibility_signals_classification_prompt(
        self, article_title: str, article_content: str
    ) -> list[dict]:

        input_text = f"Title: {article_title}\n\nContent: {article_content}"
        chat_input = [
            {
                "role": "system",
                "content": self._credibility_signals_classification_system_message,
            },
            {"role": "user", "content": input_text},
        ]

        return chat_input

    # Method to detect misinformation in an article
    def classify_article(self, state: State):
        """
        For a given article, detect whether it is credible or fake using the LLM.
        The LLM's system message sets up how it should break down the problem and strutctured output.
        """
        article_title = state["article_title"]
        article_content = state["article_content"]

        chat_input = self._generate_article_classification_prompt(
            article_title, article_content
        )

        response = self._llm.invoke(chat_input)

        state["messages"] = [response]
        return state

    def classify_article_with_credibility_signals(self, state: State):
        """
        For a given article, detect whether it is credible or fake using the LLM.
        The LLM's system message sets up how it should break down the problem and structured output.
        """
        article_title = state["article_title"]
        article_content = state["article_content"]
        credibility_signals = state.get("credibility_signals", None)

        chat_input = (
            self._generate_article_classification_prompt_with_credibility_signals(
                article_title, article_content, credibility_signals
            )
        )

        response = self._llm.invoke(chat_input)

        state["messages"] = [response]
        return state

    def detect_credibility_signals(self, state: State):
        """
        For a given article, detect specific credibility signals using the LLM.
        The LLM's system message sets up how it should break down the problem and strutctured output.
        """
        article_title = state["article_title"]
        article_content = state["article_content"]
        chat_input = self._generate_credibility_signals_classification_prompt(
            article_title=article_title, article_content=article_content
        )

        response = self._llm.invoke(chat_input)

        state["messages"] = [response]
        return state

    def _extract_signal_position(self, raw_output, signal_type):
        signal_start = raw_output.find(f"Credibility Signal: {signal_type}") + len(
            f"Credibility Signal: {signal_type}"
        )
        signal_end = raw_output.find("\n", signal_start)
        return signal_end

    def _extract_label(self, raw_output):
        label_start = raw_output.find("Label: ") + len("Label: ")
        label_end = raw_output.find("\n", label_start)
        label = raw_output[label_start:label_end].strip()
        # Convert label to an int if it's numeric
        try:
            label = float(label)
        except ValueError:
            label = None  # Handle cases where confidence is not a number

        return label, label_end

    def _extract_confidence(self, raw_output, label_end):
        confidence_start = raw_output.find("Confidence: ", label_end) + len(
            "Confidence: "
        )
        confidence_end = raw_output.find("\n", confidence_start)
        confidence = raw_output[confidence_start:confidence_end].strip()

        # Convert confidence to a float if it's numeric
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = None  # Handle cases where confidence is not a number

        return confidence, confidence_end

    def _extract_explanation(self, raw_output, confidence_end):
        explanation_start = raw_output.find("Explanation:", confidence_end) + len(
            "Explanation:"
        )
        explanation_end = min(
            raw_output.find("Credibility Signal:"),
            len(raw_output),
        )
        explanation = raw_output[explanation_start:explanation_end].strip()
        return explanation if explanation else "No explanation provided."

    def _parse_article_classification_output(self, raw_output):

        label, label_end = self._extract_label(raw_output)
        confidence, confidence_end = self._extract_confidence(raw_output, label_end)
        explanation = self._extract_explanation(raw_output, confidence_end)

        return label, confidence, explanation

    def _parse_credibility_signal_classification_output(self, raw_output):
        signals = {}
        signal_types = [
            "evidence",
            "bias",
            "inference",
            "polarising_language",
            "document_citation",
            "informal_tone",
            "explicitly_unverified_claims",
            "personal_perspective",
            "emotional_valence",
            "call_to_action",
        ]

        for signal_type in signal_types:
            signal_end = self._extract_signal_position(raw_output, signal_type)

            label, label_end = self._extract_label(raw_output[signal_end:])
            confidence, confidence_end = self._extract_confidence(
                raw_output[signal_end:], label_end
            )
            explanation = self._extract_explanation(
                raw_output[signal_end:], confidence_end
            )

            signals[signal_type] = {
                "label": label,
                "question": self._credibility_signals_prompt[signal_type],
                "confidence": confidence,
                "explanation": explanation,
            }

        return signals

    def output_article_classification(self, state: State):
        raw_output = state["messages"][-1].content

        label, confidence, explanation = self._parse_article_classification_output(
            raw_output
        )

        state["label"] = label
        state["confidence"] = confidence
        state["explanation"] = explanation

        return state

    def output_credibility_signals_classification(self, state: State):
        raw_output = state["messages"][-1].content

        signals = self._parse_credibility_signal_classification_output(raw_output)

        state["credibility_signals"] = signals

        return state
