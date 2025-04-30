import os
import json
import pandas as pd
import os
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import plotly.graph_objects as go
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def consistent_results_parser(data, chunk_number, dataset_name, experiment_id) -> dict:

    results_list = data.get("results", [])
    all_runs = []
    for result in results_list:
        run = result.get("run", {})
        example = result.get("example", {})
        eval_result = (
            result.get("evaluation_result", [])[0]
            if result.get("evaluation_result")
            else {}
        )

        # Get classification prompt safely with default empty list
        classification_prompt = run.get("outputs", {}).get("classification_prompt", [])

        # Only try to access prompt contents if we have enough elements
        system_content = (
            classification_prompt[0]["content"]
            if len(classification_prompt) > 0
            else ""
        )
        user_content = (
            classification_prompt[1]["content"]
            if len(classification_prompt) > 1
            else ""
        )

        run_data = {
            "experiment_name": data.get("experiment_name"),
            "chunk": chunk_number,
            "dataset_name": dataset_name,
            "experiment_id": experiment_id,
            "run_id": run.get("id"),
            "start_time": run.get("start_time"),
            "end_time": run.get("end_time"),
            "article_title": example.get("inputs", {}).get("article_title"),
            "article_content": example.get("inputs", {}).get("article_content"),
            "actual": example.get("outputs", {}).get("label"),
            "prediction": run.get("outputs", {}).get("label"),
            "confidence": run.get("outputs", {}).get("confidence"),
            "explanation": run.get("outputs", {}).get("explanation"),
            "eval_score": eval_result.get("score"),
            "eval_comment": eval_result.get("comment"),
            "captured_credibility_signals": run.get("outputs", {}).get(
                "credibility_signals"
            ),
            "captured_signals_critiques": run.get("outputs", {}).get(
                "signals_critiques"
            ),
            "follow_up_signals_analysis": run.get("outputs", {}).get(
                "followup_signals_analysis"
            ),
            "feature_selection": run.get("outputs", {}).get("feature_selection"),
            "classification_prompt_system": system_content,
            "classification_prompt_user": user_content,
            "classification_prompt_system_content_length": len(
                enc.encode(system_content)
            ),
            "classification_prompt_user_content_length": len(enc.encode(user_content)),
            "topic": run.get("outputs", {}).get("topic"),
        }
        all_runs.append(run_data)
    return all_runs


def split_results_parser(data, chunk_number, dataset_name, experiment_id) -> dict:
    chunk_results = data.get("results", {})
    runs = chunk_results.get("runs", [])
    examples = chunk_results.get("examples", [])
    evaluations = chunk_results.get("evaluation_results", [])
    all_runs = []
    for i, run in enumerate(runs):
        run_data = {
            "experiment_name": data.get("experiment_name"),
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
    return all_runs


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
        if type(data.get("results")) == dict:
            # If results is a dict, use split_results_parser; THIS IS THE OLD FORMAT
            runs = split_results_parser(data, chunk_number, dataset_name, experiment_id)
        else:
            # If results is a list, use consistent_results_parser
            runs = consistent_results_parser(
                data, chunk_number, dataset_name, experiment_id
            )
        all_runs.extend(runs)

    df = pd.DataFrame(all_runs)

    return df


def analyze_experiments(verbose: bool = False) -> pd.DataFrame:
    """Analyze all experiments across all datasets in the results directory."""

    results = []
    base_path = "results"

    # Walk through all dataset folders
    for dataset_folder in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        # Walk through all experiment folders in each dataset
        for experiment_id in os.listdir(dataset_path):
            if verbose:
                print(
                    f"Analyzing dataset: {dataset_folder}, experiment: {experiment_id}"
                )

            df = load_combined_results(
                dataset_name=dataset_folder, experiment_id=experiment_id
            )

            # Extract experiment details
            experiment_details = {
                "dataset": dataset_folder,
                "experiment_id": experiment_id,
                "experiment_name": df["experiment_name"].iloc[0],
                "start_time": df["start_time"].iloc[0],
            }

            # df["positive_confidence"] = df.apply(
            #     lambda x: (
            #         (1 - x["confidence"]) if x["prediction"] == 0 else x["confidence"]
            #     ),
            #     axis=1,
            # )

            df.dropna(subset=["actual", "prediction"], inplace=True)

            if len(df) == 0:
                continue

            # Calculate confusion matrix
            cm = confusion_matrix(df.actual, df.prediction)
            if verbose:
                print(f"Confusion Matrix for {experiment_id}:\n{cm}")
            if cm.shape != (2, 2):
                continue

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(df.actual, df.prediction),
                "precision": precision_score(df.actual, df.prediction),
                "recall": recall_score(df.actual, df.prediction),
                "f1": f1_score(df.actual, df.prediction),
                # "roc_auc": roc_auc_score(df.actual, df.positive_confidence),
                "true_negatives": cm[0][0],
                "false_positives": cm[0][1],
                "false_negatives": cm[1][0],
                "true_positives": cm[1][1],
            }

            experiment_results = {**experiment_details, **metrics}
            results.append(experiment_results)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    return df


def load_generated_features(dataset_name: str, experiment_id: str) -> pd.DataFrame:
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

        # If results is a list, use consistent_results_parser
        runs = consistent_results_parser(
            data, chunk_number, dataset_name, experiment_id
        )
        all_runs.extend(runs)

    df = pd.DataFrame(all_runs)
    return df
