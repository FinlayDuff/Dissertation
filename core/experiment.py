"""
Experiment Module

Handles configuration and execution of misinformation detection experiments,
including model selection, evaluation, and result logging.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import os

from langsmith.evaluation import evaluate as evaluate_remotely
from core.evaluator import evaluate_locally
from langsmith.schemas import Run, Example

from core.graph_manager import GraphManager
from core.misinformation_detection import MisinformationDetection
from config.experiments import EXPERIMENT_CONFIGS
from utils.data.langsmith_dataset import get_manager
from multiprocessing import cpu_count

from langchain_core.runnables import RunnableConfig
from tqdm.auto import tqdm
from langchain_core.callbacks.base import BaseCallbackHandler

import logging


class BatchProgressCallback(BaseCallbackHandler):
    """Callback handler for tracking batch progress"""

    def __init__(self, total):
        """Initialize progress bar with correct total"""
        self.total = total
        self.completed = 0
        self.progress_bar = tqdm(total=total, desc="Processing examples")

    def on_chain_end(self, *args, **kwargs):
        """Update only on successful completion"""
        if self.completed < self.total:  # Prevent overflow
            self.completed += 1
            self.progress_bar.update(1)

    def on_chain_error(self, *args, **kwargs):
        """Log errors but don't update progress"""
        self.progress_bar.write("❌ Error processing example")

    def __del__(self):
        """Cleanup progress bar"""
        self.progress_bar.close()
        self.progress_bar.close()


class Experiment:
    """
    Manages misinformation detection experiments with different configurations.

    Handles:
    - Experiment configuration
    - Graph execution
    - Result evaluation and logging
    - LangSmith integration
    """

    def __init__(
        self,
        experiment_name: str,
        dataset_name: str,
        description: str = "",
        chunk: bool = False,
        evaluate_locally: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize experiment with configuration.

        Args:
            experiment_name: Name of experiment configuration to use
            description: Optional description of the experiment
            verbose: Whether to print debug information
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing experiment: {dataset_name}_{experiment_name}")

        self._init_experiment_config(experiment_name)
        self._dataset_name = dataset_name
        self._description = description
        self._chunk = chunk
        self._evaluate_locally = evaluate_locally
        self._verbose = verbose

        # Initialize components
        self.detection_system = MisinformationDetection(
            verbose=verbose,
        )
        self.graph_manager = GraphManager(
            detection_system=self.detection_system,
            experiment_config=self._experiment_config,
            verbose=verbose,
        )

        # Get few-shot examples if needed
        self._few_shot_examples = (
            self.get_few_shot_examples(self._dataset_name)
            if self._experiment_config.get("few_shot")
            else None
        )

    def _init_experiment_config(self, experiment_name: str):
        """Initialize experiment configuration."""
        if experiment_name not in EXPERIMENT_CONFIGS:
            self.logger.error(f"Unknown experiment configuration: {experiment_name}")
            raise ValueError(f"Unknown experiment configuration: {experiment_name}")

        self._experiment_name = experiment_name
        self._experiment_id = str(uuid.uuid4())
        self.logger.info("Experiment ID: %s", self._experiment_id)
        self._timestamp = datetime.now().isoformat()
        self._experiment_config = EXPERIMENT_CONFIGS[experiment_name]
        self.logger.debug(f"Initialized experiment config: {self._experiment_config}")

    @staticmethod
    def get_few_shot_examples(dataset_name: str) -> str:
        """Load few-shot examples from a dataset-specific file."""
        logger = logging.getLogger(__name__)

        if not dataset_name or not isinstance(dataset_name, str):
            logger.error(f"Invalid dataset name provided: {dataset_name}")
            raise ValueError("Dataset name must be a non-empty string")

        few_shot_path = os.path.join("data", "fewshot", f"{dataset_name}.txt")
        logger.debug(f"Attempting to load few-shot examples from: {few_shot_path}")

        try:
            with open(few_shot_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                logger.debug(
                    f"Successfully loaded few-shot examples for {dataset_name}"
                )
                return content
        except FileNotFoundError:
            logger.error(f"Few-shot examples file not found: {few_shot_path}")
            raise FileNotFoundError(f"No few-shot examples found at {few_shot_path}")
        except IOError as e:
            logger.error(f"IO error reading few-shot file: {e}")
            raise IOError(f"Failed to read few-shot file: {e}")

    def _enrich_example(self, example: dict) -> dict:
        # Merge example with experiment configuration
        enriched_example = {
            **example,
            "experiment_name": self._experiment_name,
            "use_signals": self._experiment_config.get("signals", {}).get("enabled"),
            "use_bulk_signals": self._experiment_config.get("signals", {}).get(
                "use_bulk"
            ),
            "condensed_signals": self._experiment_config.get("signals", {}).get(
                "condensed"
            ),
            "few_shot": self._experiment_config.get("few_shot"),
            "few_shot_examples": self._few_shot_examples,
        }
        return enriched_example

    def _run_example(self, example: dict) -> dict:
        """Wrapper for graph execution with experiment config"""

        # Merge example with experiment configuration
        enriched_example = self._enrich_example(example)
        return self.graph_manager.run_graph_on_example(enriched_example)

    def run_evaluation(self, dataset_name: str = None, chunk_id: Optional[int] = 1):
        try:
            dataset_name = dataset_name if dataset_name else self._dataset_name

            if self._evaluate_locally:
                self.logger.info(
                    "Running local evaluation with LangGraph parallel execution"
                )
                dataset = get_manager().load_dataset(dataset_name)
                self.logger.info(f"Dataset size: {len(dataset)} examples")

                # Configure parallel execution
                num_processes = max(1, int(cpu_count() * 0.75))
                self.logger.info(f"Using {num_processes} parallel workers")

                # Create simple input-to-result mapping
                enriched_examples = [
                    self._enrich_example(example.inputs) for example in dataset
                ]
                progress_callback = BatchProgressCallback(len(enriched_examples))
                # Run evaluation in parallel using LangGraph
                results_list = self.graph_manager.graph.batch(
                    inputs=enriched_examples,
                    config=RunnableConfig(
                        max_concurrency=num_processes, callbacks=[progress_callback]
                    ),
                    return_exceptions=True,
                )
                self.logger.info(
                    f"Completed batch processing of {len(enriched_examples)} examples"
                )
                # Create parallel input-result pairs
                examples_and_results = list(zip(dataset, results_list))

                # Define lookup function
                def get_result_for_example(example):
                    # Find matching result using example identity
                    for orig_example, result in examples_and_results:
                        if (
                            orig_example.inputs["article_title"]
                            == example["article_title"]
                        ):
                            return result
                    return None

                self.logger.info(
                    "Evaluating results and applying mapping for consistency"
                )
                results = evaluate_locally(
                    dataset=dataset,
                    eval_fn=get_result_for_example,
                    evaluator=self.correct_label,
                )

            else:
                self.logger.info("Running remote evaluation via LangSmith")
                results = evaluate_remotely(
                    self._run_example,
                    data=dataset_name,
                    experiment_prefix=self._experiment_name,
                    description=self._description,
                    evaluators=[self.correct_label],
                    max_concurrency=self._experiment_config.get(
                        "max_concurrency", None
                    ),
                )

            self.logger.info(
                f"\tEvaluation completed successfully for chunk {chunk_id}"
            )
            self.log_experiment_details(results, chunk_id)

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise

    def run_evaluation_on_all_chunks(self):
        """
        Runs the evaluation across all chunks of the dataset.
        """
        langsmith_manager = get_manager()
        dataset_chunks = langsmith_manager.get_dataset_chunks(
            dataset_name=self._dataset_name
        )

        for id, chunk_name in dataset_chunks:
            self.run_evaluation(dataset_name=chunk_name, chunk_id=id)

    def run(self):
        self.logger.info(f"Starting evaluation for dataset: {self._dataset_name}")
        if self._chunk:
            self.logger.debug(
                "Experiment is configured to use the chunked dataset. Running evaluation on all chunks."
            )
            # Run evaluation on all chunks
            return self.run_evaluation_on_all_chunks()
        else:
            self.logger.debug("Running evaluation on whole dataset.")
            return self.run_evaluation()

    def log_experiment_details(self, results: Any, chunk_id: int) -> None:
        """
        Log experiment details and results to file.

        Args:
            results: Results from evaluation
        """
        self.logger.info("Logging experiment details and results")
        log_data = {
            "experiment_id": self._experiment_id,
            "timestamp": self._timestamp,
            "experiment_name": self._experiment_name,
            "description": self._description,
            "config": self._experiment_config,
            "results": self._convert_results_to_dict(results),
        }

        results_dir = os.path.join(
            os.getcwd(), "results", self._dataset_name, self._experiment_id
        )
        os.makedirs(results_dir, exist_ok=True)

        with open(
            f"{results_dir}/{chunk_id}.json",
            "w",
        ) as log_file:
            json.dump(log_data, log_file, indent=4)

    def _convert_results_to_dict(self, results: Any) -> Dict[str, Any]:
        """Convert evaluation results to serializable dictionary."""
        results_list = results._results
        output_keys_to_remove = {
            "messages",
            "article_title",
            "article_content",
            "experiment_name",
            "use_signals",
            "use_bulk_signals",
            "few_shot",
            "few_shot_examples",
        }
        return [
            {
                "run": {
                    "id": str(item["run"].id),
                    "name": item["run"].name,
                    "start_time": item["run"].start_time.isoformat(),
                    "end_time": item["run"].end_time.isoformat(),
                    "inputs": item["run"].inputs,
                    "outputs": {
                        k: v
                        for k, v in item["run"].outputs.items()
                        if k not in output_keys_to_remove
                    },
                },
                "example": {
                    "inputs": item["example"].inputs,
                    "outputs": item["example"].outputs,
                },
                "evaluation_result": [
                    {
                        "key": eval_result.key,
                        "score": eval_result.score,
                        "value": eval_result.value,
                        "comment": eval_result.comment,
                        "correction": eval_result.correction,
                    }
                    for eval_result in item["evaluation_results"]["results"]
                ],
            }
            for item in results_list
        ]

    @staticmethod
    def correct_label(root_run: Run, example: Example) -> dict:
        """Correct label evaluator for LangSmith."""
        predicted = root_run.outputs.get("label", "")
        actual = example.outputs.get("label")

        return {
            "score": 1.0 if predicted == actual else 0.0,
            "comment": f"Predicted: {predicted}, Actual: {actual}",
        }
