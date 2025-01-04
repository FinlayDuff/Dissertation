# experiment.py

import json
import uuid
from datetime import datetime
from typing_extensions import TypedDict
from typing import Annotated
from core.graph_manager import (
    DirectClassificationGraphManager,
    StagedClassificationGraphManager,
)
from core.prompts import get_prompt
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
import os


class Experiment:
    def __init__(
        self,
        article_classification_prompt_name: Annotated[
            str, "The prompt name for article classification"
        ],
        llm_model_name: Annotated[str, "The name of the llm model to be used"],
        dataset_name: Annotated[str, "The name of the dataset being evaluated"],
        description: Annotated[str, "Description of the experiment"] = "",
        experiment_type: Annotated[
            str, "Type of the experiment"
        ] = "direct_classification",
        credibility_signals_prompt_name: Annotated[
            str,
            "The prompt name for credibility signals. This defaults but can be overridden is desired. _credibility_signals_prompt_content is only set if experiment_type is 'staged_classification'",
        ] = "credibility_signals",
        verbose: bool = False,
    ):
        self._verbose = verbose

        self._article_classification_prompt_name = article_classification_prompt_name
        self._article_classification_prompt_content = get_prompt(
            article_classification_prompt_name
        )

        self._llm_model_name = llm_model_name

        self._dataset_name = dataset_name
        self._description = description
        self._experiment_type = experiment_type
        if self._experiment_type == "staged_classification":
            self._credibility_signals_prompt_name = credibility_signals_prompt_name
            self._credibility_signals_prompt_content = get_prompt(
                credibility_signals_prompt_name
            )

        self._experiment_id = str(uuid.uuid4())
        self._timestamp = datetime.now().isoformat()
        self._initialise_experiment_name()
        self.graph_manager = self.initialize_graph_manager()

        if self._verbose:
            print(f"Initialized Experiment: {self._experiment_name}")

    def _initialise_experiment_name(self):
        self._experiment_name = f"{self._experiment_type}_{self._article_classification_prompt_name}_{self._llm_model_name}_{self._dataset_name}"

    def initialize_graph_manager(self):
        if self._experiment_type == "direct_classification":
            return DirectClassificationGraphManager(
                self._llm_model_name,
                self._article_classification_prompt_content,
                verbose=self._verbose,
            )
        elif self._experiment_type == "staged_classification":
            return StagedClassificationGraphManager(
                self._llm_model_name,
                self._article_classification_prompt_content,
                self._credibility_signals_prompt_content,
                verbose=self._verbose,
            )
        else:
            raise ValueError(f"Unknown experiment type: {self._experiment_type}")

    def convert_results_to_dict(self, results):
        # Extract relevant data from results._results
        results_list = results._results
        aggregated_results = {"runs": [], "examples": [], "evaluation_results": []}

        for item in results_list:
            run_data = {
                "id": str(item["run"].id),
                "name": item["run"].name,
                "start_time": item["run"].start_time.isoformat(),
                "end_time": item["run"].end_time.isoformat(),
                "inputs": item["run"].inputs,
                "outputs": item["run"].outputs,
            }
            example_data = {
                "inputs": item["example"].inputs,
                "outputs": item["example"].outputs,
            }
            evaluation_data = [
                {
                    "key": eval_result.key,
                    "score": eval_result.score,
                    "value": eval_result.value,
                    "comment": eval_result.comment,
                    "correction": eval_result.correction,
                }
                for eval_result in item["evaluation_results"]["results"]
            ]

            aggregated_results["runs"].append(run_data)
            aggregated_results["examples"].append(example_data)
            aggregated_results["evaluation_results"].extend(evaluation_data)

        return aggregated_results

    def log_experiment_details(self, results):

        log_data = {
            "experiment_id": self._experiment_id,
            "timestamp": self._timestamp,
            "prompt_name": self._article_classification_prompt_name,
            "model_name": self._llm_model_name,
            "dataset_name": self._dataset_name,
            "description": self._description,
            "experiment_name": self._experiment_name,
            "results": self.convert_results_to_dict(results),
            "credibility_signals": self.graph_manager.stored_credibility_signals,
        }

        # Use an absolute path for the results directory
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)

        with open(
            f"results/{self._timestamp}_experiment_{self._experiment_id}.json", "w"
        ) as log_file:
            json.dump(log_data, log_file, indent=4)

    @staticmethod
    def correct_label(root_run: Run, example: dict) -> dict:
        predicted_label = root_run.outputs.get("label")
        actual_label = example.outputs.get("label")
        score = predicted_label == actual_label
        return {"score": int(score), "key": "correct_label"}

    def run_evaluation(self):
        # Use the evaluate function to run the evaluation
        results = evaluate(
            self.graph_manager.run_graph_on_example,
            data=self._dataset_name,
            evaluators=[self.correct_label],
            experiment_prefix=self._experiment_name,
            description=self._description,
        )

        self.log_experiment_details(results)
        return results
