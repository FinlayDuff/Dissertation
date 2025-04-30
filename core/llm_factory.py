from dataclasses import dataclass
from typing import Optional, Any, Tuple, Dict, TypedDict, List, Union
from config.experiments import EXPERIMENT_CONFIGS, ModelConfig
from config.prompts import TASK_PROMPTS, STRUCTURED_OUTPUTS, USER_INPUT_PROMPT
from config.signals import CREDIBILITY_SIGNALS, CREDIBILITY_SIGNALS_CONDENSED
import json

from core.state import State
from utils.langchain.llm_model_selector import get_llm_from_model_name
import re

from enum import Enum
from utils.constants import (
    POLARITY,
    LABEL,
    CONFIDENCE,
    RELEVANCE,
    TRUST,
)


class TaskType(Enum):
    CLASSIFY_ARTICLE = "classify_article"
    BULK_SIGNALS = "bulk_signals"
    INDIVIDUAL_SIGNAL = "individual_signal"
    CRITIC = "critic"
    SIGNAL_CRITIC = "signal_critic"
    FOLLOWUP_ANALYSIS = "followup_analysis"
    FOLLOWUP_RAG = "followup_rag"
    FOLLOWUP_CORROBORATION = "followup_corroboration"
    FEATURE_SELECTOR = "feature_selector"
    FOLLOWUP_CLAIM_EXTRACTOR = "followup_claim_extractor"
    FOLLOWUP_CLAIM_VERIFICATION = "followup_claim_verification"


class ClassificationResult(TypedDict):
    """
    Result of an article classification.

    Attributes:
        label: Classification result ("Credible" or "Fake")
        confidence: Confidence score between 0 and 1
        explanation: Detailed reasoning for classification
    """

    label: str
    confidence: float
    explanation: str


class SignalResult(TypedDict):
    """
    Result of a single credibility signal analysis.

    Attributes:
        label: Signal classification result
        confidence: Confidence score between 0 and 1
        explanation: Detailed reasoning for signal detection
    """

    label: str
    confidence: float
    explanation: str


class BulkSignalsResult(TypedDict):
    """
    Collection of multiple credibility signal results.

    Attributes:
        signals: Dictionary mapping signal names to their analysis results
    """

    signals: Dict[str, SignalResult]


@dataclass
class LLMResponse:
    """
    Wrapper for LLM response containing both raw and parsed content.

    Attributes:
        raw_content: Original response string from the LLM
        parsed_content: Structured data parsed from the response
        input_content: Original input content sent to the LLM
    """

    raw_content: str
    parsed_content: Optional[
        Union[ClassificationResult, SignalResult, BulkSignalsResult]
    ] = None
    input_content: Optional[str] = None


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for LLM initialization and operation.

    Attributes:
        task_prompt: Instructions given to the LLM
        model_name: Name of the model to use
        temperature: Sampling temperature (0-1)
        local: Whether to run model locally
        timeout: Maximum time in seconds for response
        structured_output: Expected JSON schema for output
    """

    task_prompt: str
    task_type: TaskType
    model_name: str
    temperature: float = 0.0
    local: bool = False
    timeout: int = 300
    structured_output: Optional[str] = None
    user_content: Optional[str] = None

    @classmethod
    def from_model_config(
        cls,
        model_config: Dict[str, Any],
        task_prompt: str,
        task_type: TaskType,
        structured_output: Optional[str] = None,
        user_content: Optional[str] = None,
    ):

        return cls(
            task_prompt=task_prompt,
            task_type=task_type,
            model_name=model_config["model_name"],
            temperature=model_config["temperature"],
            local=model_config["local"],
            timeout=model_config["timeout"],
            structured_output=structured_output,
            user_content=user_content,
        )


SCRATCH_RE = re.compile(r"<!--scratch-->.*?<!--scratch-->", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n([\s\S]*?)\n\s*```\s*$", re.IGNORECASE)


class LLMFactory:
    @staticmethod
    def create_for_node(node_name: str, state: State) -> Tuple[Any, str]:
        """Creates appropriate LLM configuration based on node and state"""
        if node_name == "classify_article":
            if state.get("credibility_signals"):
                return LLMFactory._create_classification_with_signals(state)
            if state.get("few_shot"):
                return LLMFactory._create_few_shot_classification(state)
            return LLMFactory._create_zero_shot_classification(state)

        if node_name == "detect_credibility_signals":
            if state.get("use_bulk_signals"):
                return LLMFactory._create_bulk_signals_detection(state)
            return LLMFactory._create_individual_signals_detection(state)

        if node_name == "critic_article_classification":
            return LLMFactory._create_article_classification_critic(state)
        if node_name == "critic_signal_classification":
            return LLMFactory._create_signal_classification_critic(state)

        if node_name == "followup_analysis":
            return LLMFactory._create_followup_signal_analysis(state)
        if node_name == "followup_rag":
            return LLMFactory._create_follow_up_rag(state)

        if node_name == "feature_selector":
            return LLMFactory._create_feature_selector(state)

        raise ValueError(f"Unknown node type: {node_name}")

    @staticmethod
    def _get_model_config(state: State, model_type: str) -> ModelConfig:
        """Get model config from experiment setup or fall back to defaults"""
        experiment_name = state.get("experiment_name")
        if experiment_name and experiment_name in EXPERIMENT_CONFIGS:
            return EXPERIMENT_CONFIGS[experiment_name][f"{model_type}_model"]
        raise ValueError(f"No configuration found for experiment {experiment_name}")

    @staticmethod
    def _create_zero_shot_classification(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "classification")

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["zero_shot_classification"],
            task_type=TaskType.CLASSIFY_ARTICLE,
            structured_output=STRUCTURED_OUTPUTS["article_classification"],
            user_content=USER_INPUT_PROMPT["base_inputs"].format(
                title=state["article_title"],
                content=state["article_content"],
            ),
        )

        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_few_shot_classification(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "classification")

        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )

        examples = state.get("few_shot_examples", [])
        user_content += USER_INPUT_PROMPT["few_shot_extension"].format(
            few_shot_block=examples
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["few_shot_classification"],
            task_type=TaskType.CLASSIFY_ARTICLE,
            structured_output=STRUCTURED_OUTPUTS["article_classification"],
            user_content=user_content,
        )

        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_individual_signals_detection(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "signals")

        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )

        signal_type = state.get("current_signal")
        # Use condensed signals for bulk detection
        if state.get("condensed_signals"):
            signal_questions = CREDIBILITY_SIGNALS_CONDENSED
        else:
            signal_questions = CREDIBILITY_SIGNALS
        user_content += USER_INPUT_PROMPT["individual_signal_extension"].format(
            signal_type=signal_type, signal_config=signal_questions[signal_type]
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["individual_signal"],
            task_type=TaskType.INDIVIDUAL_SIGNAL,
            structured_output=STRUCTURED_OUTPUTS["individual_signal"],
            user_content=f"""
                    Title: {state['article_title']}\n
                    Content: {state['article_content']}
                """,
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_bulk_signals_detection(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "signals")
        task_prompt = TASK_PROMPTS["bulk_signals"]

        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )

        # Use condensed signals for bulk detection
        if state.get("condensed_signals"):
            signal_questions = CREDIBILITY_SIGNALS_CONDENSED
        else:
            signal_questions = CREDIBILITY_SIGNALS

        user_content += USER_INPUT_PROMPT["bulk_signals_extension"].format(
            signals_list=json.dumps(signal_questions, indent=2)
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=task_prompt,
            task_type=TaskType.BULK_SIGNALS,
            structured_output=STRUCTURED_OUTPUTS["bulk_signals"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config, state=state)

    @staticmethod
    def signal_weight(row, overall_trust=1):
        return (
            POLARITY[row["polarity"]]
            * LABEL[row["label"]]
            * CONFIDENCE[row["confidence"]]
            * RELEVANCE[row["relevance"]]
            * TRUST[overall_trust]
        )

    @staticmethod
    def format_signals_data_for_classification(
        sig_dict: dict, condensed_signals: bool
    ) -> list:
        signal_questions = (
            CREDIBILITY_SIGNALS_CONDENSED if condensed_signals else CREDIBILITY_SIGNALS
        )

        signals_data_for_classification = {
            signal_name: {
                "label": signal_data["label"],
                "polarity": signal_questions.get(signal_name, {}).get("polarity"),
                "confidence": signal_data["confidence"],
                "explanation": signal_data["explanation"],
            }
            for signal_name, signal_data in sig_dict.items()
        }

        return signals_data_for_classification

    @staticmethod
    def format_signals_data_for_critic(sig_dict: dict, condensed_signals: bool) -> list:
        signal_questions = (
            CREDIBILITY_SIGNALS_CONDENSED if condensed_signals else CREDIBILITY_SIGNALS
        )

        signals_data_for_classification = {
            signal_name: {
                "signal_type": signal_questions.get(signal_name, {}).get("signal_type"),
                "question": signal_questions.get(signal_name, {}).get("question"),
                "polarity": signal_questions.get(signal_name, {}).get("polarity"),
                **signal_data,
            }
            for signal_name, signal_data in sig_dict.items()
        }
        return signals_data_for_classification

    @staticmethod
    def format_signal_critic_data_for_classification(
        sig_dict: dict,
    ) -> list:

        signals_data_for_classification = {
            signal_name: {
                "label": signal_data["label"],
                "polarity": signal_data["polarity"],
                "quality": signal_data["quality"],
                "relevance": signal_data["relevance"],
                "explanation": signal_data["explanation"],
            }
            for signal_name, signal_data in sig_dict["keep"].items()
        }
        return signals_data_for_classification

    @staticmethod
    def format_selected_features_for_classification(
        sig_dict: dict, condensed_signals: bool
    ) -> list:
        signal_questions = (
            CREDIBILITY_SIGNALS_CONDENSED if condensed_signals else CREDIBILITY_SIGNALS
        )
        signals_data_for_classification = {
            signal_name: {
                "question": signal_questions.get(signal_name, {}).get("question"),
                "label": signal_data["label"],
                "confidence": signal_data["confidence"],
                "explanation": signal_data["explanation"],
            }
            for signal_name, signal_data in sig_dict.items()
        }
        return signals_data_for_classification

    @staticmethod
    def _create_classification_with_signals(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "classification")
        task_prompt = TASK_PROMPTS["zero_shot_classification_signals"]

        # Parse the state to get the signals
        signals = state.get("credibility_signals", {})
        condensed_signals = state.get("condensed_signals")
        signals_data_for_classification = (
            LLMFactory.format_signals_data_for_classification(
                signals, condensed_signals
            )
        )
        signals_critiques = state.get("signals_critiques", {})
        followup_analysis = state.get("followup_signals_analysis", {})
        selected_features = state.get("feature_selection", {})

        # Always inject this
        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )

        if not selected_features:
            if signals_data_for_classification and not signals_critiques:
                user_content += USER_INPUT_PROMPT[
                    "classified_signals_extension"
                ].format(
                    signals_list=json.dumps(signals_data_for_classification, indent=2)
                )

            if signals_critiques:
                critc_list = LLMFactory.format_signal_critic_data_for_classification(
                    signals_critiques
                )
                user_content += USER_INPUT_PROMPT["critic_extension"].format(
                    critic_list=json.dumps(critc_list, indent=2),
                    overall_extraction_trust=signals_critiques["overall_trust"],
                )
                task_prompt = TASK_PROMPTS["zero_shot_classification_signals_critic"]

            if followup_analysis:
                user_content += USER_INPUT_PROMPT[
                    "followup_analysis_classification_extension"
                ].format(followup_analysis=json.dumps(followup_analysis, indent=2))
                task_prompt = TASK_PROMPTS[
                    "zero_shot_classification_signals_critic_followup"
                ]

        examples = state.get("few_shot_examples", [])
        if examples:
            task_prompt = TASK_PROMPTS["few_shot_classification_signals"]
            user_content += USER_INPUT_PROMPT["few_shot_extension"].format(
                few_shot_block=examples
            )

        elif selected_features:
            signals_data_for_classification = (
                LLMFactory.format_selected_features_for_classification(
                    selected_features["signals"], condensed_signals
                )
            )
            task_prompt = TASK_PROMPTS["zero_shot_classification_feature_selector"]
            user_content += USER_INPUT_PROMPT[
                "classification_selected_features_extension"
            ].format(
                selected_features=json.dumps(signals_data_for_classification, indent=2),
            )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=task_prompt,
            task_type=TaskType.CLASSIFY_ARTICLE,
            structured_output=STRUCTURED_OUTPUTS["article_classification"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_article_classification_critic(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "critic")

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["critic_article_classification"],
            task_type=TaskType.CRITIC,
            structured_output=STRUCTURED_OUTPUTS["critic"],
            user_content=f"""
                    Title: {state['article_title']}\n
                    Content: {state['article_content']}
                """,
        )

        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_signal_classification_critic(state: State) -> Tuple[Any, str]:
        # This is consistent no matter what's being critiqued
        model_config = LLMFactory._get_model_config(state, "critic")
        signals = state.get("credibility_signals", {})
        condensed_signals = state.get("condensed_signals")
        signals_data_for_critic = LLMFactory.format_signals_data_for_critic(
            signals, condensed_signals
        )
        topic = state.get("topic", "unknown")

        user_content = USER_INPUT_PROMPT[
            "signal_classification_critic_extension"
        ].format(
            topic=topic,
            signals_list=json.dumps(signals_data_for_critic, indent=2),
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["signal_classification_critic"],
            task_type=TaskType.SIGNAL_CRITIC,
            structured_output=STRUCTURED_OUTPUTS["signal_classification_critic"],
            user_content=user_content,
        )

        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_followup_signal_analysis(state: State):
        """
        Creates LLM configuration for detailed follow-up analysis of a specific credibility signal.

        Args:
            state: Current state containing article and signal information
            signal_name: Name of the signal being analyzed
        """
        model_config = LLMFactory._get_model_config(state, "followup")

        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )

        signal_type = state.get("current_signal")
        signal_classification = state.get("credibility_signals", {}).get(
            signal_type, {}
        )
        critic_explanation = (
            state.get("signals_critiques", {}).get("follow_up", {}).get(signal_type)
        )

        user_content += USER_INPUT_PROMPT["followup_analysis_signals_extension"].format(
            signal_type=signal_type,
            signal_classification=json.dumps(signal_classification, indent=2),
            critic_explanation=critic_explanation,
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["followup_analysis"],
            task_type=TaskType.FOLLOWUP_ANALYSIS,
            structured_output=STRUCTURED_OUTPUTS["followup_analysis"],
            user_content=user_content,
        )

        return LLMFactory._initialize_llm(config)

    def _create_follow_up_rag(state: State) -> Any:
        model_config = LLMFactory._get_model_config(state, "rag")
        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )
        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["followup_rag"],
            task_type=TaskType.FOLLOWUP_RAG,
            structured_output=STRUCTURED_OUTPUTS["followup_rag"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config)

    def create_follow_up_corroboration(
        state: State, core_fact: str, retrieved_articles: List[dict]
    ) -> Any:
        model_config = LLMFactory._get_model_config(state, "followup")
        user_content = USER_INPUT_PROMPT["followup_corroboration"].format(
            title=state["article_title"],
            core_fact=core_fact,
            retrieved_articles=json.dumps(retrieved_articles, indent=2),
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["followup_corroboration"],
            task_type=TaskType.FOLLOWUP_CORROBORATION,
            structured_output=STRUCTURED_OUTPUTS["followup_corroboration"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config)

    def create_claim_extractor(
        state: State,
    ) -> Any:
        model_config = LLMFactory._get_model_config(state, "followup")
        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["followup_claim_extractor"],
            task_type=TaskType.FOLLOWUP_CLAIM_EXTRACTOR,
            structured_output=STRUCTURED_OUTPUTS["followup_claim_extractor"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config)

    def create_claim_verification(state: State, claims: List[dict]) -> Any:
        model_config = LLMFactory._get_model_config(state, "followup")
        user_content = USER_INPUT_PROMPT["followup_claim_verification"].format(
            claims_list=json.dumps(claims, indent=2),
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["followup_claim_verification"],
            task_type=TaskType.FOLLOWUP_CLAIM_VERIFICATION,
            structured_output=STRUCTURED_OUTPUTS["followup_claim_verification"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config)

    def _create_feature_selector(state: State):
        model_config = LLMFactory._get_model_config(state, "critic")
        user_content = USER_INPUT_PROMPT["base_inputs"].format(
            title=state["article_title"],
            content=state["article_content"],
        )
        credibility_signals = state.get("credibility_signals", "No signals found")
        critic = state.get("signals_critiques", "No critiques found")
        followup_analysis = state.get(
            "followup_signals_analysis", "No followup analysis found"
        )

        user_content += USER_INPUT_PROMPT["feature_selector_extension"].format(
            signals_list=json.dumps(credibility_signals, indent=2),
            critic_notes=json.dumps(critic, indent=2),
            followup_analysis=json.dumps(followup_analysis, indent=2),
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["feature_selector"],
            task_type=TaskType.FEATURE_SELECTOR,
            structured_output=STRUCTURED_OUTPUTS["feature_selector"],
            user_content=user_content,
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _parse_classification(content: str):
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)

            # Check if it has "label", "confidence", "explanation" keys
            if all(key in data for key in ["label", "confidence", "explanation"]):
                # If label valid classification
                if data["label"] in ["REAL", "FAKE"]:
                    return {
                        "label": 1 if data["label"] == "REAL" else 0,
                        "confidence": data["confidence"],
                        "explanation": data["explanation"],
                    }

        except json.JSONDecodeError:
            # If it isn't valid JSON
            return None

    @staticmethod
    def _parse_critic(content: str):
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)

            # Check if it has "label", "explanation" keys
            if all(key in data for key in ["label", "explanation"]):
                # If label is an int
                if isinstance(data["label"], str):
                    return {
                        "critic_decision": data["label"],
                        "critic_explanation": data["explanation"],
                    }

        except json.JSONDecodeError:
            # If it isn't valid JSON
            return None

    @staticmethod
    def _parse_signals_critic(content: str):
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            critic_output = json.loads(content)

            if not critic_output:
                print("Failed to parse content due to missing critic output", content)
                return None
            # Check each signal has required fields
            if not all(
                key in critic_output.keys()
                for key in [
                    "keep",
                    "follow_up",
                    "overall_trust",
                    "critic_notes",
                ]
            ):
                print(
                    "Failed to parse content due to missing keys", critic_output.items()
                )
                return None

            return {"signals_critiques": critic_output}

        except json.JSONDecodeError:
            print("Failed to parse content due JSON error", content)
            return None

    @staticmethod
    def _parse_signal(content: str) -> SignalResult:
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()

        try:
            data = json.loads(content)
            missing_keys = [
                key
                for key in ["label", "confidence", "explanation", "article_excerpts"]
                if key not in data
            ]
            # Check if it has "label", "confidence", "explanation" keys
            if missing_keys:
                print("Failed to parse content due to missing keys", missing_keys)
                return None

            else:
                return {
                    "label": data["label"],
                    "article_excerpts": data["article_excerpts"],
                    "confidence": data["confidence"],
                    "explanation": data["explanation"],
                }

        except json.JSONDecodeError:
            # If it isn't valid JSON
            return None

    @staticmethod
    def _parse_bulk_signals(content: str, condensed: bool) -> BulkSignalsResult:
        # Matches a leading ``` or ```json (any capitalisation) and the trailing ```
        fence = re.match(
            r"^\s*```(?:json)?\s*\n([\s\S]*?)\n\s*```\s*$", content, flags=re.IGNORECASE
        )
        if fence:
            content = fence.group(1)  # keep only the inner JSON
        try:
            data = json.loads(content)
            signals = data.get("signals", {})
            if condensed:
                signal_questions = CREDIBILITY_SIGNALS_CONDENSED
            else:
                signal_questions = CREDIBILITY_SIGNALS
            # Check that all required credibility signals are present
            if not all(signal in signals for signal in signal_questions):
                print("Failed to parse content due to missing signals", content)
                return None

            # Check each signal has required fields
            for signal_name, signal_data in signals.items():
                if not all(
                    key in signal_data for key in ["label", "confidence", "explanation"]
                ):
                    print("Failed to parse content due to missing keys", content)
                    return None

            return {"signals": signals}

        except json.JSONDecodeError:
            print("Failed to parse content due to faulty JSON", content)
            return None

    def _parse_rag(content: str) -> dict:
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)
            missing_keys = [key for key in ("core_fact", "queries") if key not in data]
            if missing_keys:
                print("Failed to parse content due to missing keys", missing_keys)
                return None
            return data

        except json.JSONDecodeError:
            print("Failed to parse content due to faulty JSON", content)
            return None

    def _parse_corroboration(content: str) -> dict:
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)
            missing_keys = [
                key
                for key in (
                    "corroborating_articles",
                    "reputable_corroboration_of_source",
                    "explanation",
                )
                if key not in data
            ]
            if missing_keys:
                print("Failed to parse content due to missing keys", missing_keys)
                return None
            return data

        except json.JSONDecodeError:
            print("Failed to parse content due to faulty JSON", content)
            return None

    @staticmethod
    def _parse_claim_extractor(content: str) -> dict:
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)

            if not isinstance(data, list):
                print("Failed to parse content due to not being a list")
                return None
            else:
                # Check each claim has required fields
                for claim in data:
                    missing_keys = [
                        key
                        for key in (
                            "claim",
                            "importance",
                            "evidence",
                        )
                        if key not in claim
                    ]
                    if missing_keys:
                        print(
                            "Failed to parse content due to missing keys", missing_keys
                        )
                        return None
            return data

        except json.JSONDecodeError:
            print("Failed to parse content due to faulty JSON", content)
            return None

    @staticmethod
    def _parse_claim_verification(content: str) -> dict:
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)
            missing_keys = [
                key
                for key in (
                    "claims",
                    "explicitly_unverified_claims",
                    "explanation",
                )
                if key not in data
            ]
            if missing_keys:
                print("Failed to parse content due to missing keys", missing_keys)
                return None
            return data

        except json.JSONDecodeError:
            print("Failed to parse content due to faulty JSON", content)
            return None

    def _parse_feature_selector(content: str) -> dict:
        # only match non-scratch content
        content = SCRATCH_RE.sub("", content).strip()
        # check for injection json formatting instructions by LLM
        m = FENCE_RE.match(content)
        if m:
            content = m.group(1).strip()
        try:
            data = json.loads(content)
            missing_keys = [key for key in ("signals",) if key not in data]
            if missing_keys:
                print("Failed to parse content due to missing keys", missing_keys)
                return None
            return data

        except json.JSONDecodeError:
            print("Failed to parse content due to faulty JSON", content)
            return None

    @staticmethod
    def _initialize_llm(
        config: LLMConfig, state: Optional[State] = None
    ) -> Tuple[Any, str]:
        """Initialize LLM with configuration and parsing."""

        llm = get_llm_from_model_name(config)

        # Create a subclass that wraps the invoke method
        class WrappedLLM(type(llm)):
            def invoke(self):

                message = [
                    {
                        "role": "system",
                        "content": config.task_prompt + config.structured_output,
                    }
                ] + [
                    {
                        "role": "user",
                        "content": config.user_content,
                    },
                ]
                response = super().invoke(message)

                if hasattr(response, "content"):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    raise ValueError(f"Unexpected response type: {type(response)}")

                # Use task type enum for parsing
                if config.task_type == TaskType.CLASSIFY_ARTICLE:
                    parsed = LLMFactory._parse_classification(content)
                elif config.task_type == TaskType.BULK_SIGNALS:
                    parsed = LLMFactory._parse_bulk_signals(
                        content, state.get("condensed_signals")
                    )
                elif config.task_type == TaskType.INDIVIDUAL_SIGNAL:
                    parsed = LLMFactory._parse_signal(content)
                elif config.task_type == TaskType.CRITIC:
                    parsed = LLMFactory._parse_critic(content)
                elif config.task_type == TaskType.SIGNAL_CRITIC:
                    parsed = LLMFactory._parse_signals_critic(content)
                elif config.task_type == TaskType.FOLLOWUP_ANALYSIS:
                    parsed = LLMFactory._parse_signal(content)
                elif config.task_type == TaskType.FOLLOWUP_RAG:
                    parsed = LLMFactory._parse_rag(content)
                elif config.task_type == TaskType.FOLLOWUP_CORROBORATION:
                    parsed = LLMFactory._parse_corroboration(content)
                elif config.task_type == TaskType.FEATURE_SELECTOR:
                    parsed = LLMFactory._parse_feature_selector(content)
                elif config.task_type == TaskType.FOLLOWUP_CLAIM_EXTRACTOR:
                    parsed = LLMFactory._parse_claim_extractor(content)
                elif config.task_type == TaskType.FOLLOWUP_CLAIM_VERIFICATION:
                    parsed = LLMFactory._parse_claim_verification(content)
                else:
                    raise ValueError(f"Unknown task type: {config.task_type}")

                return LLMResponse(
                    raw_content=content, parsed_content=parsed, input_content=message
                )

        return WrappedLLM(**llm.__dict__)
