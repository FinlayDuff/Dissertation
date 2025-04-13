from tqdm import tqdm
from typing import List, Callable, Any
from datetime import datetime
import uuid


class MockRun:
    def __init__(self, inputs, outputs, name="local-eval"):
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
    """
    Locally evaluate a dataset using the experiment's graph.

    Args:
        dataset: List of Example objects with .inputs and .outputs
        eval_fn: Function that runs the graph (takes .inputs, returns prediction)
        evaluator: Function that scores the prediction (takes run + example)

    Returns:
        An object mimicking the LangSmith EvaluationResult with ._results
    """
    results = []
    for example in tqdm(dataset, desc="Local Evaluation"):
        # Run the model/graph
        outputs = eval_fn(example.inputs)

        run = MockRun(example.inputs, outputs)

        # Call the evaluator with mock run + real example
        evaluation_result = evaluator(run, example)
        evaluation_result["key"] = evaluator.__name__
        evaluation_result["correction"] = None
        evaluation_result["value"] = run.outputs.get("label", "")

        # Package result
        results.append(
            {
                "example": example,
                "run": run,
                "evaluation_results": {"results": evaluation_result},
            }
        )

    # Wrap in object with ._results like LangSmith returns
    class LocalResults:
        def __init__(self, results):
            self._results = results

    return LocalResults(results)
