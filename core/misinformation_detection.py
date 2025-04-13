"""
Misinformation Detection Module

This module provides a system for detecting misinformation in articles using LLMs.
It supports multiple detection strategies including direct classification,
credibility signal analysis, and multi-stage verification.

The system can:
- Classify articles as credible or fake
- Detect credibility signals (bias, emotion, etc.)
- Make decisions about additional analysis needs
- Handle both bulk and individual signal detection
"""

import json

from core.llm_factory import LLMFactory
from core.state import State
from config.signals import CREDIBILITY_SIGNALS
from utils.langchain.llm_model_selector import retry_on_rate_limit
from core.followup_analysis_tools import FOLLOWUP_TOOLS


class MisinformationDetection:
    """
    A system for detecting misinformation in articles using Large Language Models.

    This class orchestrates the process of analyzing articles for misinformation
    through various methods including direct classification, credibility signal
    detection, and multi-stage verification.

    The system maintains state through a State object that tracks article content,
    classification results, credibility signals, and LLM responses.
    """

    def __init__(self, verbose: bool = False):
        """Initialize the misinformation detection system."""
        self.verbose = verbose

    @retry_on_rate_limit()
    def classify_article(self, state: State) -> State:
        """
        Classify an article as credible or fake using an LLM.

        Args:
            state: Current state containing:
                - article_title: Title of the article
                - article_content: Content to classify
                - experiment_name: Name of experiment config to use
                - few_shot: Whether to use few-shot examples
                - few_shot_examples: Optional list of examples

        Returns:
            Updated state containing:
                - messages: List of LLM responses
                - label: Classification result ("Credible" or "Fake")
                - confidence: Confidence score (0-1)
                - explanation: Reasoning for classification

        Raises:
            ValueError: If required state fields are missing
        """
        llm = LLMFactory.create_for_node("classify_article", state)
        response = llm.invoke()

        state["messages"] = [response.raw_content]
        if response.parsed_content:
            state.update(response.parsed_content)
        return state

    @retry_on_rate_limit()
    def detect_signals(self, state: State) -> State:
        """
        Detect credibility signals either in bulk or individually.

        Args:
            state: Current state containing:
                - article_title: Title of the article
                - article_content: Content to analyze
                - use_bulk_signals: Whether to detect all signals in one call
                - experiment_name: Name of experiment config to use

        Returns:
            Updated state containing:
                - messages: List of LLM responses
                - credibility_signals: Dictionary of detected signals
        """
        if state.get("use_bulk_signals"):
            return self._detect_bulk_signals(state)
        return self._detect_individual_signals(state)

    def _detect_bulk_signals(self, state: State) -> State:
        """
        Detect all credibility signals in one LLM call.

        Args:
            state: Current state with article information

        Returns:
            Updated state with bulk signals detection results
        """
        llm = LLMFactory.create_for_node("detect_credibility_signals", state)

        response = llm.invoke()

        state["messages"] = [response.raw_content]
        if response.parsed_content:
            if self.verbose:
                print("credibility_signals parsed successfully")
            state["credibility_signals"] = response.parsed_content.get("signals", {})
        return state

    def _detect_individual_signals(self, state: State) -> State:
        """
        Detect each credibility signal with separate LLM calls.

        Args:
            state: Current state with article information

        Returns:
            Updated state with individual signal detection results
        """
        signals = {}
        messages = []

        for signal_type in CREDIBILITY_SIGNALS:
            state["current_signal"] = signal_type
            llm = LLMFactory.create_for_node("detect_credibility_signals", state)

            response = llm.invoke()

            messages.append(response.raw_content)
            if response.parsed_content:
                signals[signal_type] = response.parsed_content

        state["messages"] = messages
        state["credibility_signals"] = signals

        return state

    @retry_on_rate_limit()
    def make_critic_decision(self, state: State) -> State:
        """
        Decide if additional analysis is needed based on current results.

        Args:
            state: Current state containing:
                - label: Current classification
                - confidence: Classification confidence
                - credibility_signals: Detected signals

        Returns:
            Updated state containing:
                - critic_decision: "YES" if more analysis needed, "NO" otherwise
        """
        llm = LLMFactory.create_for_node("critic_decision", state)

        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "classification": {
                                "label": state.get("label"),
                                "confidence": state.get("confidence"),
                            },
                            "signals": state.get("credibility_signals", {}),
                        }
                    ),
                },
            ]
        )

        state["critic_decision"] = response.raw_content
        if response.parsed_content:
            state.update(response.parsed_content)
        return state

    def critic_signal_classification(self, state: State) -> State:
        """
        Detect all credibility signals in one LLM call.

        Args:
            state: Current state with article information

        Returns:
            Updated state with bulk signals detection results
        """
        llm = LLMFactory.create_for_node("critic_signal_classification", state)

        response = llm.invoke()

        state["messages"] = [response.raw_content]
        if response.parsed_content:
            state["signals_critiques"] = response.parsed_content.get(
                "signals_critiques", {}
            )
        return state

    def run_followup_analysis(self, state: State) -> State:
        """
        For each credibility signal that can call a tool the method selects the appropriate pre-trained model to run (e.g. bias classifier,
        RAG for source analysis) and updates the state accordingly.

        Returns:
            str: State with updated follow-up analysis results
        """
        signals_critiques = state.get("signals_critiques", {})

        followup_results = {}
        for signal_name in signals_critiques:
            # Only process signals that are flagged for review
            if (
                signals_critiques[signal_name]["label"] == "TRUE"
                and signal_name in FOLLOWUP_TOOLS
            ):
                if self.verbose:
                    print(
                        f"Running followup analysis on credibility signal: {signal_name}"
                    )

                # Determine the appropriate tool for the signal.
                followup_tool = FOLLOWUP_TOOLS[signal_name]
                if followup_tool["method"] != "llm":
                    # Run the follow-up tool; assume it updates and returns the state.
                    result = followup_tool["method"](state)
                    followup_results[signal_name] = result
                elif followup_tool["method"] == "llm":
                    state["current_signal"] = signal_name
                    llm = LLMFactory.create_for_node("followup_analysis", state)

                    response = llm.invoke()

                    state["messages"] = [response.raw_content]
                    if response.parsed_content:
                        response.parsed_content["analysis_type"] = "llm"
                        followup_results[signal_name] = response.parsed_content
                else:
                    if self.verbose:
                        print(f"No follow-up tool configured for signal: {signal_name}")
        if followup_results:
            state["followup_signals_analysis"] = followup_results
        return state
