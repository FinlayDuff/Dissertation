from typing_extensions import TypedDict
from typing import Annotated
from utils.langchain.prompts import (
    NAIVE_ZERO_SHOT_CLASSIFICATION_PROMPT,
    ROBUST_ZERO_SHOT_CLASSIFICATION_PROMPT,
)


class Experiment:
    def __init__(
        self,
        prompt_name: Annotated[str, "The prompt used to tell the LLM what to do"],
        model_name: Annotated[str, "The name of the experiment"],
        dataset_name: Annotated[str, "The name of the dataset being evaluated"],
    ):
        self.prompt_name = prompt_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.initialise_experiment_name()

    def initialise_experiment_name(self):
        self.experiment_name = f"""
            {self.prompt_name}_{self.model_name}_{self.dataset_name}
        """
