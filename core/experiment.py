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

from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

from core.graph_manager import GraphManager
from core.misinformation_detection import MisinformationDetection
from config.experiments import EXPERIMENT_CONFIGS, ExperimentConfig


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
        self, experiment_name: str, description: str = "", verbose: bool = False
    ):
        """
        Initialize experiment with configuration.

        Args:
            experiment_name: Name of experiment configuration to use
            description: Optional description of the experiment
            verbose: Whether to print debug information
        """
        if experiment_name not in EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown experiment configuration: {experiment_name}")

        self._experiment_id = str(uuid.uuid4())
        self._timestamp = datetime.now().isoformat()
        self._experiment_name = experiment_name
        self._description = description
        self._verbose = verbose

        # Initialize components
        self.detection_system = MisinformationDetection(
            verbose=verbose,
        )
        self.graph_manager = GraphManager(
            detection_system=self.detection_system,
            verbose=verbose,
        )

    def run_evaluation(self, dataset_name: str):
        """
        Run the experiment on a dataset.

        Args:
            dataset_name: Name of dataset to evaluate

        Returns:
            Evaluation results from LangSmith
        """
        # Get experiment configuration
        experiment_config = EXPERIMENT_CONFIGS[self._experiment_name]

        def evaluate_example(example: dict) -> dict:
            """Wrapper for graph execution with experiment config"""
            # Merge example with experiment configuration
            enriched_example = {
                **example,
                "experiment_name": self._experiment_name,
                "use_signals": experiment_config.get("signals").get("enabled"),
                "use_bulk_signals": experiment_config.get("signals").get("use_bulk"),
                "few_shot": experiment_config.get("few_shot"),
                "few_shot_examples": experiment_config.get("few_shot_examples"),
            }

            return self.graph_manager.run_graph_on_example(enriched_example)

        results = evaluate(
            evaluate_example,
            data=dataset_name,
            experiment_prefix=self._experiment_name,
            description=self._description,
            evaluators=[self.correct_label],
            max_concurrency=experiment_config.get("max_concurrency", None),
        )

        self.log_experiment_details(results)
        return results

    def log_experiment_details(self, results: Any) -> None:
        """
        Log experiment details and results to file.

        Args:
            results: Results from evaluation
        """
        experiment_config = EXPERIMENT_CONFIGS[self._experiment_name]

        log_data = {
            "experiment_id": self._experiment_id,
            "timestamp": self._timestamp,
            "experiment_name": self._experiment_name,
            "description": self._description,
            "config": experiment_config,
            "results": self._convert_results_to_dict(results),
            "credibility_signals": self.graph_manager.get_stored_signals(),
        }

        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)

        with open(
            f"results/{self._timestamp}_experiment_{self._experiment_id}.json", "w"
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
