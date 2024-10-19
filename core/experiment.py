# experiment.py

import json
import uuid
from datetime import datetime
from typing_extensions import TypedDict
from typing import Annotated
from core.graph_manager import (
    BasicClassificationGraphManager,
    MultiPromptGraphManager,
)
from core.prompts import get_prompt
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run
import os


class Experiment:
    def __init__(
        self,
        prompt_name: Annotated[str, "The prompt used to tell the LLM what to do"],
        model_name: Annotated[str, "The name of the experiment"],
        dataset_name: Annotated[str, "The name of the dataset being evaluated"],
        description: Annotated[str, "Description of the experiment"] = "",
        experiment_type: Annotated[
            str, "Type of the experiment"
        ] = "basic_classification",
    ):
        self.prompt_name = prompt_name
        self.prompt_content = get_prompt(prompt_name)  # Fetch the prompt content
        if not self.prompt_content:
            raise ValueError(f"Prompt '{prompt_name}' not found in PROMPTS.")
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.description = description
        self.experiment_type = experiment_type
        self.experiment_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.initialise_experiment_name()
        self.graph_manager = self.initialize_graph_manager()

    def initialise_experiment_name(self):
        self.experiment_name = f"{self.experiment_type}_{self.prompt_name}_{self.model_name}_{self.dataset_name}"

    def initialize_graph_manager(self):
        if self.experiment_type == "basic_classification":
            return BasicClassificationGraphManager(self.model_name, self.prompt_content)
        elif self.experiment_type == "multi_prompt":
            return MultiPromptGraphManager(self.model_name, self.prompt_content)
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

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
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "prompt_name": self.prompt_name,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "description": self.description,
            "experiment_name": self.experiment_name,
            "results": self.convert_results_to_dict(results),
        }
        # Use an absolute path for the results directory
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)

        with open(f"results/experiment_{self.experiment_id}.json", "w") as log_file:
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
            data=self.dataset_name,
            evaluators=[self.correct_label],
            experiment_prefix=self.experiment_name,
            description=self.description,
        )

        self.log_experiment_details(results)
        return results
