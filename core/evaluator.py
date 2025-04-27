from tqdm import tqdm
from typing import List, Callable, Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class EvaluationResult:
    """Standardized evaluation result structure"""

    key: str
    score: float
    value: Any
    comment: Optional[str] = None
    correction: Optional[Any] = None


class MockRun:
    """Mimics LangSmith Run structure"""

    def __init__(self, inputs: Dict, outputs: Dict, name: str = "local-eval"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.inputs = inputs
        self.outputs = outputs


def evaluate_locally(
    dataset: List[Any],
    eval_fn: Callable[[dict], dict],
    evaluator: Callable[[dict, Any], dict],
) -> Any:
    """Local evaluation with standardized result structure"""
    results = []
    for example in tqdm(dataset, desc="Local Evaluation"):
        # Run prediction
        outputs = eval_fn(example.inputs)
        run = MockRun(example.inputs, outputs)

        # Get evaluation score and comment
        eval_dict = evaluator(run, example)

        # Create standardized result
        evaluation_result = EvaluationResult(
            key=evaluator.__name__,
            score=eval_dict["score"],
            value=run.outputs.get("label", ""),
            comment=eval_dict.get("comment"),
            correction=None,
        )

        # Package in LangSmith-like structure
        results.append(
            {
                "example": example,
                "run": run,
                "evaluation_results": {"results": [evaluation_result]},
            }
        )

    # Wrap in object with ._results like LangSmith returns
    class LocalResults:
        def __init__(self, results):
            self._results = results

    return LocalResults(results)
