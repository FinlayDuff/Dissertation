from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
from typing import Callable, Dict
from core.state import State
import yaml
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import logging
import random
from torch.multiprocessing import Lock
import threading

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Ensure cudnn operates deterministically
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)


def load_model_config():
    """Load model configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config" / "hugging_face_models.yml"
    logger.debug(f"Loading model config from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["models"]
            logger.debug(f"Successfully loaded {len(config)} model configurations")
            return config
    except FileNotFoundError:
        logger.error(f"Model config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing model config file: {e}")
        raise


# Cache the config at module level
MODEL_CONFIG = load_model_config()


class Classifier:
    _thread_local = threading.local()

    def __init__(self, signal_name: str):
        self.logger = logging.getLogger(f"{__name__}.{signal_name}")
        self.logger.debug(f"Initializing classifier for signal: {signal_name}")
        self.signal_name = signal_name
        self.model_config = MODEL_CONFIG.get(signal_name)
        # Create a per-instance lock to serialize pipeline calls per thread
        self._model_lock = threading.Lock()
        self._load_model()
        self._create_pipeline()

    def _load_model(self):
        """Load the model and tokenizer based on the signal name."""
        if not self.model_config:
            self.logger.error(f"No model configuration found for {self.signal_name}")
            raise ValueError(f"No model configuration found for {self.signal_name}")

        try:
            self.logger.debug(f"Loading tokenizer for {self.model_config['name']}")
            # Use fast tokenizer may cause issues with multithreading; force use_fast=False if needed.
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["name"],
                local_files_only=True,
            )

            self.logger.debug(f"Loading model for {self.model_config['name']}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config["name"], local_files_only=True
            )
            # Set the model to evaluation mode to disable dropout
            self.model.eval()
            self.logger.debug(
                f"Successfully loaded model and tokenizer for {self.signal_name}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model/tokenizer: {e}", exc_info=True)
            raise

    def _create_pipeline(self):
        """Create a pipeline for the model. Stored in thread-local storage."""
        self.logger.debug("Creating classification pipeline")
        if not hasattr(self._thread_local, "pipe"):
            try:
                self._thread_local.pipe = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # Set to GPU device id if available (e.g., 0) or keep -1 for CPU.
                    batch_size=8,
                    truncation=True,
                    padding=True,
                    max_length=self.model_config["max_length"],
                    top_k=None,
                )
                self.logger.debug("Successfully created classification pipeline")
            except Exception as e:
                self.logger.error("Failed to create pipeline", exc_info=True)
                raise
        return self._thread_local.pipe

    def _classify(self, input_text):
        """Classify the article content using the loaded model."""
        self.logger.debug("Starting text classification")
        self.logger.debug(f"Tokenizing input text of length: {len(input_text)}")
        inputs = self.tokenizer(
            input_text,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=self.model_config["max_length"],
            stride=128,
            return_tensors=None,
        )

        chunks = [
            self.tokenizer.decode(input_ids, skip_special_tokens=True)
            for input_ids in inputs["input_ids"]
        ]
        chunk_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
        self.logger.debug(f"Split text into {len(chunks)} chunks")

        self.logger.debug("Running predictions on chunks")
        pipe = self._create_pipeline()
        with self._model_lock:
            predictions = pipe(chunks, batch_size=8)
        self.logger.debug(f"Completed classification for {len(chunks)} chunks")
        return predictions, chunk_lengths

    @staticmethod
    def _get_overall_classification(predictions, weights):
        """
        Calculate weighted average scores per label across chunks, then select the winning label.
        Expected predictions: list of lists of dictionaries, each with "label" and "score" keys.
        Expected weights: list of token counts corresponding to each chunk.
        """
        label_scores = defaultdict(list)
        label_weights = defaultdict(list)

        for pred, w in zip(predictions, weights):
            for entry in pred:
                label_scores[entry["label"]].append(entry["score"])
                label_weights[entry["label"]].append(w)

        avg_scores = {}
        for label in label_scores:
            avg_scores[label] = float(
                np.average(label_scores[label], weights=label_weights[label])
            )

        overall_label = max(avg_scores, key=avg_scores.get)
        overall_score = avg_scores[overall_label]
        return overall_label, overall_score

    def _format_output(self, label, avg_score):
        label_map = self.model_config.get("label_mapping", {})
        if label in label_map:
            label = label_map[label]
        return {
            "analysis_type": self.model_config["description"],
            "result": {
                "label": label,
                "confidence": avg_score,
            },
        }

    @staticmethod
    def _fetch_input_text(state):
        return state["article_content"]

    def __call__(self, state):
        self.logger.debug(f"Processing {self.signal_name} classification request")
        try:
            input_text = self._fetch_input_text(state)
            predictions, weights = self._classify(input_text)
            overall_label, overall_score = self._get_overall_classification(
                predictions, weights
            )
            result = self._format_output(overall_label, overall_score)
            self.logger.debug(
                f"Classification complete: {result['result']['label']} ({result['result']['confidence']:.2f})"
            )
            return result
        except Exception as e:
            self.logger.error(f"Classification failed: {e}", exc_info=True)
            raise


class TitleClassifier(Classifier):
    def __init__(self, signal_name: str):
        super().__init__(signal_name)

    @staticmethod
    def _fetch_input_text(state):
        return state["article_title"]


logging.info("Initializing followup analysis tools")
FOLLOWUP_TOOLS: Dict[str, Callable[[State], Dict]] = {
    "bias": {
        "method": Classifier("bias"),
        "description": MODEL_CONFIG["bias"]["description"],
    },
    "call_to_action": {
        "method": Classifier("call_to_action"),
        "description": MODEL_CONFIG["call_to_action"]["description"],
    },
    "clickbait": {
        "method": TitleClassifier("clickbait"),
        "description": MODEL_CONFIG["clickbait"]["description"],
    },
    "document_citation": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "emotional_valence": {
        "method": Classifier("emotional_valence"),
        "description": MODEL_CONFIG["emotional_valence"]["description"],
    },
    "evidence": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "explicitly_unverified_claims": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "expert_citation": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "impoliteness": {
        "method": Classifier("impoliteness"),
        "description": MODEL_CONFIG["impoliteness"]["description"],
    },
    "incivility": {
        "method": Classifier("incivility"),
        "description": MODEL_CONFIG["incivility"]["description"],
    },
    "incorrect_spelling": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "inference": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "informal_tone": {
        "method": Classifier("informal_tone"),
        "description": MODEL_CONFIG["informal_tone"]["description"],
    },
    "misleading_about_content": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "personal_perspective": {
        "method": Classifier("personal_perspective"),
        "description": MODEL_CONFIG["personal_perspective"]["description"],
    },
    "polarising_language": {
        "method": Classifier("polarising_language"),
        "description": MODEL_CONFIG["polarising_language"]["description"],
    },
    "reported_by_other_sources": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "sensationalism": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "source_credibility": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "topic": {
        "method": Classifier("topic"),
        "description": MODEL_CONFIG["topic"]["description"],
    },
    "evidence_present": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "inference_error": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "credible_sourcing": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "external_corroboration": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "strong_framing_tone": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
    "writing_quality_alert": {
        "method": "llm",
        "description": "An LLM prompted to reclassify the credibility signal",
    },
}
