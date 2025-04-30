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
from utils.langchain.llm_model_selector import retry_on_api_exceptions
from core.followup_analysis_tools import FOLLOWUP_TOOLS, Classifier, TitleClassifier
from core.rag import (
    retrieve_from_search,
    remove_exact_matching_results,
    remove_credibility_sources,
    retrieve_from_news,
)


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
        if response.input_content:
            state["classification_prompt"] = response.input_content
        return state

    @retry_on_api_exceptions()
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

    @retry_on_api_exceptions()
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
        Quality-control gate between the signal-extractor and the final REAL/FAKE classifier.

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
        followup_signals = state.get("signals_critiques", {}).get("follow_up", [])

        followup_results = {}
        for signal_name in followup_signals:
            # Only process signals if they have a valid follow-up tool configured.
            if signal_name in FOLLOWUP_TOOLS:
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

    def classify_topic(self, state: State) -> State:
        """
        Sets state['topic'] to a short lowercase label
        labels_list=['education', 'human interest', 'society', 'sport', 'crime, law and justice',
        'disaster, accident and emergency incident', 'arts, culture, entertainment and media', 'politics',
        'economy, business and finance', 'lifestyle and leisure', 'science and technology',
        'health', 'labour', 'religion', 'weather', 'environment', 'conflict, war and peace'],
        """
        article_topic_classifier = FOLLOWUP_TOOLS["topic"]
        topic_classifier_output = article_topic_classifier["method"](state)
        topic = topic_classifier_output.get("result", {}).get("label", None)
        if topic is None:
            raise ValueError("Topic classification failed.")
        state["topic"] = topic
        return state

    def followup_classifier(self, state: State) -> State:
        followup_signals = state.get("signals_critiques", {}).get("follow_up", [])
        classifier_based = [
            signal_name
            for signal_name in followup_signals
            if isinstance(
                FOLLOWUP_TOOLS.get(signal_name, {}).get("method"),
                (Classifier, TitleClassifier),
            )
        ]
        results = {}
        for signal_name in classifier_based:
            tool = FOLLOWUP_TOOLS[signal_name]["method"]
            results[signal_name] = tool(state)
        if "followup_signals_analysis" not in state:
            state["followup_signals_analysis"] = {}
        state["followup_signals_analysis"].update(results)
        return state

    def followup_llm(self, state: State) -> State:
        followup_signals = state.get("signals_critiques", {}).get("follow_up", [])
        llm_based = [
            signal_name
            for signal_name in followup_signals
            if FOLLOWUP_TOOLS.get(signal_name, {}).get("method") == "llm"
        ]

        results = {}
        for signal_name in llm_based:
            if signal_name == "external_corroboration":
                results[signal_name] = self.corroboration_rag_chain(state)
            elif signal_name == "explicitly_unverified_claims":
                results[signal_name] = self.explicitly_unverified_claims_chain(state)
            else:
                results[signal_name] = self.basic_llm_followup(state, signal_name)

        if "followup_signals_analysis" not in state:
            state["followup_signals_analysis"] = {}
        state["followup_signals_analysis"].update(results)
        return state

    def basic_llm_followup(self, state: State, signal_name: str) -> State:
        state["current_signal"] = signal_name
        llm = LLMFactory.create_for_node("followup_analysis", state)
        response = llm.invoke()

        state["messages"] = [response.raw_content]
        if response.parsed_content:
            response.parsed_content["analysis_type"] = "llm"
            return response.parsed_content

    def corroboration_rag_chain(self, state: State) -> State:

        # First, we need to summarize the article content and generate search queries.
        # This is done using the "followup_rag" LLM.
        llm = LLMFactory.create_for_node("followup_rag", state)
        response = llm.invoke()
        if response.parsed_content:
            core_fact = response.parsed_content.get("core_fact", "")
            search_queries = response.parsed_content.get("queries", [])

            # Now we can use the summary and search queries to retrieve relevant articles.
            retrieved_articles = retrieve_from_news(queries=search_queries)

            # Filter out near-exact matches to the original article.
            filtered_articles = remove_exact_matching_results(
                title=state["article_title"],
                start=state["article_content"][:200],
                results=retrieved_articles,
            )

            filtered_articles = remove_credibility_sources(filtered_articles)

            # Finally, we can use the summary and retrieved articles to perform the follow-up analysis.
            # This is done using the "followup_corroboration" LLM.
            llm = LLMFactory.create_follow_up_corroboration(
                state, core_fact=core_fact, retrieved_articles=filtered_articles
            )
            response = llm.invoke()
            if response.parsed_content:
                response.parsed_content["analysis_type"] = "RAG_chain_of_thought"
                return response.parsed_content

    def feature_selector(self, state: State) -> State:
        """
        Use an LLM to read the full set of signals, critiques, follow-ups,
        extractor trust, topic, and a snippet of the article, and then OUTPUT
        a *condensed* JSON containing only:
        - article.title
        - article.snippet
        - topic
        - extractor_trust
        - a shortlist of the top 3-5 signals (with label, polarity, relevance, one-line reason)
        - any follow-ups the LLM deems critical
        """
        llm = LLMFactory.create_for_node("feature_selector", state)
        response = llm.invoke()
        if response.parsed_content:
            state["feature_selection"] = response.parsed_content
        return state

    def explicitly_unverified_claims_chain(self, state: State) -> State:
        """
        Use the LLM to analyze the credibility of sources in the article.
        """
        llm = LLMFactory.create_claim_extractor(state)
        response = llm.invoke()
        if response.parsed_content:
            claims = response.parsed_content
            for claim in claims:
                retrieved_articles = retrieve_from_search(queries=claim["claim"])
                filtered_articles = remove_exact_matching_results(
                    title=state["article_title"],
                    start=state["article_content"][:200],
                    results=retrieved_articles,
                )
                filtered_articles = remove_credibility_sources(filtered_articles)

                claim["retreived_articles"] = filtered_articles

            llm = LLMFactory.create_claim_verification(state, claim)
            response = llm.invoke()
            if response.parsed_content:
                response.parsed_content["analysis_type"] = "RAG_chain_of_thought"
                return response.parsed_content

    def credible_sources(self, state: State) -> State:
        """
        Use the LLM to analyze the credibility of sources in the article.
        """
        llm = LLMFactory.create_source_extraction("credible_sources", state)
        response = llm.invoke()
        if response.parsed_content:
            state["credible_sources"] = response.parsed_content
        return state
