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

    def _init_experiment_config(self, experiment_name: str):
        """
        Initialize experiment configuration.

        Raises:
            ValueError: If experiment_name is not found in configurations
        """
        if experiment_name not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown experiment configuration: {experiment_name}")
        self._experiment_name = experiment_name
        self._experiment_id = str(uuid.uuid4())
        self._timestamp = datetime.now().isoformat()
        self._experiment_config = EXPERIMENT_CONFIGS[experiment_name]

    @staticmethod
    def get_few_shot_examples(dataset_name: str) -> str:
        """
        Load few-shot examples from a dataset-specific file.

        Args:
            dataset_name: Name of the dataset to load few-shot examples for

        Returns:
            String containing few-shot examples

        Raises:
            ValueError: If dataset_name is empty or invalid
            FileNotFoundError: If few-shot examples file does not exist
            IOError: If there is an error reading the file
        """
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError("Dataset name must be a non-empty string")

        few_shot_path = os.path.join("data", "fewshot", f"{dataset_name}.txt")

        try:
            with open(few_shot_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"No few-shot examples found at {few_shot_path}")
        except IOError as e:
            raise IOError(f"Failed to read few-shot file: {e}")

    def run_evaluation(
        self,
        dataset_name: str = None,
        chunk_id: Optional[int] = 1,
    ):
        """
        Run the experiment on a dataset.

        Args:
            dataset_name: Name of dataset to evaluate

        Returns:
            Evaluation results from LangSmith
        """

        few_shot_examples = None
        if self._experiment_config.get("few_shot"):
            few_shot_examples = self.get_few_shot_examples(self._dataset_name)

        def evaluate_example(example: dict) -> dict:
            """Wrapper for graph execution with experiment config"""

            # Merge example with experiment configuration
            enriched_example = {
                **example,
                "experiment_name": self._experiment_name,
                "use_signals": self._experiment_config.get("signals", {}).get(
                    "enabled"
                ),
                "use_bulk_signals": self._experiment_config.get("signals", {}).get(
                    "use_bulk"
                ),
                "few_shot": self._experiment_config.get("few_shot"),
                "few_shot_examples": few_shot_examples,
            }

            return self.graph_manager.run_graph_on_example(enriched_example)

        dataset_name = dataset_name if dataset_name else self._dataset_name

        if self._evaluate_locally:
            # Load dataset examples from LangSmith
            dataset = get_manager().load_dataset(dataset_name)
            results = evaluate_locally(
                dataset=dataset,
                eval_fn=evaluate_example,
                evaluator=self.correct_label,
            )
        else:
            import logging

            logging.getLogger("langsmith").setLevel(logging.ERROR)
            results = evaluate_remotely(
                evaluate_example,
                data=dataset_name,
                experiment_prefix=self._experiment_name,
                description=self._description,
                evaluators=[self.correct_label],
                max_concurrency=self._experiment_config.get("max_concurrency", None),
            )

        self.log_experiment_details(results, chunk_id)

    def run_evaluation_on_all_chunks(self):
        """
        Runs the evaluation across all chunks of the dataset.
        """
        langsmith_manager = get_manager()
        dataset_chunks = langsmith_manager.get_dataset_chunks(
            dataset_name=self._dataset_name
        )

        for id, chunk_name in dataset_chunks:
            print(f"Running experiment on chunk: {chunk_name}")
            self.run_evaluation(dataset_name=chunk_name, chunk_id=id)

    def run(self):

        if self._chunk:
            print(
                "[INFO] Experiment is configured to use the chunked dataset. Running evaluation on all chunks."
            )
            # Run evaluation on all chunks
            return self.run_evaluation_on_all_chunks()
        else:
            print("[INFO] Running evaluation on whole dataset.")
            return self.run_evaluation()

    def log_experiment_details(self, results: Any, chunk_id: int) -> None:
        """
        Log experiment details and results to file.

        Args:
            results: Results from evaluation
        """

        log_data = {
            "experiment_id": self._experiment_id,
            "timestamp": self._timestamp,
            "experiment_name": self._experiment_name,
            "description": self._description,
            "config": self._experiment_config,
            "results": self._convert_results_to_dict(results),
            "credibility_signals": self.graph_manager.get_stored_signals(),
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
        return {
            "runs": [
                {
                    "id": str(item["run"].id),
                    "name": item["run"].name,
                    "start_time": item["run"].start_time.isoformat(),
                    "end_time": item["run"].end_time.isoformat(),
                    "inputs": item["run"].inputs,
                    "outputs": item["run"].outputs,
                }
                for item in results_list
            ],
            "examples": [
                {
                    "inputs": item["example"].inputs,
                    "outputs": item["example"].outputs,
                }
                for item in results_list
            ],
            "evaluation_results": [
                {
                    "key": eval_result.key,
                    "score": eval_result.score,
                    "value": eval_result.value,
                    "comment": eval_result.comment,
                    "correction": eval_result.correction,
                }
                for item in results_list
                for eval_result in item["evaluation_results"]["results"]
            ],
        }

    @staticmethod
    def correct_label(root_run: Run, example: Example) -> dict:
        """Correct label evaluator for LangSmith."""
        predicted = root_run.outputs.get("label", "")
        actual = example.outputs.get("label")

        return {
            "score": 1.0 if predicted == actual else 0.0,
            "comment": f"Predicted: {predicted}, Actual: {actual}",
        }
