import os
import json
import pandas as pd


def load_combined_results(dataset_name: str, experiment_id: str) -> pd.DataFrame:
    """
    Load all chunked results for a given dataset and experiment ID,
    and return a combined DataFrame.

    Args:
        dataset_name (str): Name of the dataset (folder name under results/)
        experiment_id (str): UUID or identifier used for the experiment run

    Returns:
        pd.DataFrame: Combined DataFrame of all runs in the experiment
    """
    experiment_path = os.path.join("results", dataset_name, experiment_id)
    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(f"No such experiment directory: {experiment_path}")

    # Load only numeric .json files and sort them by chunk number
    chunk_files = sorted(
        [
            f
            for f in os.listdir(experiment_path)
            if f.endswith(".json") and f[:-5].isdigit()
        ],
        key=lambda f: int(f[:-5]),
    )

    all_runs = []

    for file in chunk_files:
        chunk_path = os.path.join(experiment_path, file)
        with open(chunk_path, "r") as f:
            data = json.load(f)

        chunk_number = int(file[:-5])
        chunk_results = data.get("results", {})
        runs = chunk_results.get("runs", [])
        examples = chunk_results.get("examples", [])
        evaluations = chunk_results.get("evaluation_results", [])

        for i, run in enumerate(runs):
            run_data = {
                "chunk": chunk_number,
                "dataset_name": dataset_name,
                "experiment_id": experiment_id,
                "run_id": run.get("id"),
                "start_time": run.get("start_time"),
                "end_time": run.get("end_time"),
                "article_title": run.get("inputs", {})
                .get("example", {})
                .get("article_title"),
                "article_content": run.get("inputs", {})
                .get("example", {})
                .get("article_content"),
                "actual": examples[i].get("outputs", {}).get("label"),
                "prediction": run.get("outputs", {}).get("label"),
                "confidence": run.get("outputs", {}).get("confidence"),
                "explanation": run.get("outputs", {}).get("explanation"),
            }

            # Attach corresponding evaluation score and comment if available
            if i < len(evaluations):
                run_data.update(
                    {
                        "eval_score": evaluations[i].get("score"),
                        "eval_comment": evaluations[i].get("comment"),
                    }
                )

            all_runs.append(run_data)

    df = pd.DataFrame(all_runs)
    return df
