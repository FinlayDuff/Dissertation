from dataclasses import dataclass
from typing import Optional, Any, Tuple, Dict, TypedDict, List, Union
from config.experiments import EXPERIMENT_CONFIGS, ModelConfig
from config.prompts import TASK_PROMPTS, STRUCTURED_OUTPUTS
from config.signals import CREDIBILITY_SIGNALS
import json

from core.state import State
from utils.langchain.llm_model_selector import get_llm_from_model_name

from enum import Enum


class TaskType(Enum):
    CLASSIFY_ARTICLE = "classify_article"
    BULK_SIGNALS = "bulk_signals"
    INDIVIDUAL_SIGNAL = "individual_signal"
    CRITIC = "critic"


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
    """

    raw_content: str
    parsed_content: Optional[
        Union[ClassificationResult, SignalResult, BulkSignalsResult]
    ] = None


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
        few_shot_examples: List of example input/output pairs
    """

    task_prompt: str
    task_type: TaskType
    model_name: str
    temperature: float = 0.0
    local: bool = False
    timeout: int = 300
    structured_output: Optional[str] = None
    few_shot_examples: Optional[Tuple[Dict[str, str], ...]] = (
        None  # Changed from List to Tuple
    )

    @classmethod
    def from_model_config(
        cls,
        model_config: Dict[str, Any],
        task_prompt: str,
        task_type: TaskType,
        structured_output: Optional[str] = None,
    ):
        # Convert any mutable types to immutable
        few_shot = model_config.get("few_shot_examples")
        if few_shot:
            few_shot = tuple(tuple(x.items()) for x in few_shot)

        return cls(
            task_prompt=task_prompt,
            task_type=task_type,
            model_name=model_config["model_name"],
            temperature=model_config["temperature"],
            local=model_config["local"],
            timeout=model_config["timeout"],
            structured_output=structured_output,
            few_shot_examples=few_shot,
        )


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

        if node_name == "critic_decision":
            return LLMFactory._create_critic(state)

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
        )

        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_few_shot_classification(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "classification")
        examples = state.get("few_shot_examples", [])
        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["few_shot_classification"],
            task_type=TaskType.CLASSIFY_ARTICLE,
            structured_output=STRUCTURED_OUTPUTS["article_classification"],
        )
        config.few_shot_examples = examples
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_individual_signals_detection(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "signals")
        signal_type = state.get("current_signal")
        signal_config = CREDIBILITY_SIGNALS[signal_type]

        task_prompt = TASK_PROMPTS["individual_signal"].format(
            signal_type=signal_type, signal_config=signal_config
        )

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=task_prompt,
            task_type=TaskType.INDIVIDUAL_SIGNAL,
            structured_output=STRUCTURED_OUTPUTS["individual_signal"],
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_bulk_signals_detection(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "signals")
        task_prompt = TASK_PROMPTS["bulk_signals"].format(
            signals_list=json.dumps(CREDIBILITY_SIGNALS, indent=2)
        )
        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=task_prompt,
            task_type=TaskType.BULK_SIGNALS,
            structured_output=STRUCTURED_OUTPUTS["bulk_signals"],
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_classification_with_signals(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "classification")

        signals = state.get("credibility_signals", {})
        signals_data_for_classification = {
            signal_name: {
                **signal_data,
                "prompt": CREDIBILITY_SIGNALS.get(signal_name, {}).get("prompt"),
            }
            for signal_name, signal_data in signals.items()
        }

        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=f"{TASK_PROMPTS['zero_shot_classification']}\nUse these credibility signals:\n{json.dumps(signals_data_for_classification, indent=2)}",
            task_type=TaskType.CLASSIFY_ARTICLE,
            structured_output=STRUCTURED_OUTPUTS["article_classification"],
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _create_critic(state: State) -> Tuple[Any, str]:
        model_config = LLMFactory._get_model_config(state, "critic")
        config = LLMConfig.from_model_config(
            model_config,
            task_prompt=TASK_PROMPTS["critic"],
            task_type=TaskType.CRITIC,
            structured_output=STRUCTURED_OUTPUTS["critic"],
        )
        return LLMFactory._initialize_llm(config)

    @staticmethod
    def _parse_classification(content: str):
        try:
            data = json.loads(content)

            # Check if it has "label", "confidence", "explanation" keys
            if all(key in data for key in ["label", "confidence", "explanation"]):
                # If label is an int
                if isinstance(data["label"], int):
                    return {
                        "label": data["label"],
                        "confidence": float(data["confidence"]),
                        "explanation": data["explanation"],
                    }

        except json.JSONDecodeError:
            # If it isn't valid JSON
            return None

    @staticmethod
    def _parse_critic(content: str):
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
    def _parse_signal(content: str) -> SignalResult:
        try:
            data = json.loads(content)
            # Check if it has "label", "confidence", "explanation" keys
            if all(key in data for key in ["label", "confidence", "explanation"]):
                # If label is an int
                if isinstance(data["label"], int):
                    return {
                        "label": data["label"],
                        "confidence": float(data["confidence"]),
                        "explanation": data["explanation"],
                    }

        except json.JSONDecodeError:
            # If it isn't valid JSON
            return None

    @staticmethod
    def _parse_bulk_signals(content: str) -> BulkSignalsResult:
        try:
            data = json.loads(content)
            signals = data.get("signals", {})

            # Check that all required credibility signals are present
            if not all(signal in signals for signal in CREDIBILITY_SIGNALS):
                return None

            # Check each signal has required fields
            for signal_name, signal_data in signals.items():
                if not all(
                    key in signal_data for key in ["label", "confidence", "explanation"]
                ):
                    return None
                # Convert confidence to float if needed
                signals[signal_name]["confidence"] = float(signal_data["confidence"])

            return {"signals": signals}

        except json.JSONDecodeError:
            return None

    @staticmethod
    def _initialize_llm(config: LLMConfig) -> Tuple[Any, str]:
        """Initialize LLM with configuration and parsing."""

        llm = get_llm_from_model_name(config)

        # Create a subclass that wraps the invoke method
        class WrappedLLM(type(llm)):
            def invoke(self, user_input):

                system_prompt = config.task_prompt + config.structured_output
                message = [{"role": "system", "content": system_prompt}] + user_input
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
                    parsed = LLMFactory._parse_bulk_signals(content)
                elif config.task_type == TaskType.INDIVIDUAL_SIGNAL:
                    parsed = LLMFactory._parse_signal(content)
                elif config.task_type == TaskType.CRITIC:
                    parsed = LLMFactory._parse_critic(content)
                else:
                    raise ValueError(f"Unknown task type: {config.task_type}")

                return LLMResponse(raw_content=content, parsed_content=parsed)

        return WrappedLLM(**llm.__dict__)
