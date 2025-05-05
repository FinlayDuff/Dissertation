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
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
import hashlib

ALPHA = 0.05
N_PERM = 10_000


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
            "article_id": hashlib.sha256(
                example.get("inputs", {}).get("article_title").encode("utf-8")
            ).hexdigest()[:12],
            "article_title": example.get("inputs", {}).get("article_title"),
            "article_content": example.get("inputs", {}).get("article_content"),
            "article_title_length": len(
                enc.encode(example.get("inputs", {}).get("article_title", ""))
            ),
            "article_content_length": len(
                enc.encode(example.get("inputs", {}).get("article_content", ""))
            ),
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

            df["elapsed_sec"] = (
                pd.to_datetime(df["end_time"], utc=True, errors="coerce")
                - pd.to_datetime(df["start_time"], utc=True, errors="coerce")
            ).dt.total_seconds()

            mean_s = df["elapsed_sec"].mean()
            p50_s = df["elapsed_sec"].quantile(0.50)
            p99_s = df["elapsed_sec"].quantile(0.99)

            # df["positive_confidence"] = df.apply(
            #     lambda x: (
            #         (1 - x["confidence"]) if x["prediction"] == 0 else x["confidence"]
            #     ),
            #     axis=1,
            # )

            mean_classification_prompt_length = None
            if "classification_prompt_user_content_length" in df.columns:
                mean_classification_prompt_length = df[
                    "classification_prompt_user_content_length"
                ].mean()

            df.dropna(subset=["actual", "prediction"], inplace=True)

            if len(df) == 0:
                continue

            # Calculate confusion matrix
            cm = confusion_matrix(df.actual, df.prediction, labels=[0, 1])
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
                "accuracy": accuracy_score(df.actual, df.prediction),
                "precision_real": precision_score(
                    df.actual, df.prediction, pos_label=1
                ),
                "recall_real": recall_score(df.actual, df.prediction, pos_label=1),
                "f1_real": f1_score(df.actual, df.prediction, pos_label=1),
                "precision_fake": precision_score(
                    df.actual, df.prediction, pos_label=0
                ),
                "recall_fake": recall_score(df.actual, df.prediction, pos_label=0),
                "f1_fake": f1_score(df.actual, df.prediction, pos_label=0),
                "f1_macro": f1_score(df.actual, df.prediction, average="macro"),
                # "roc_auc": roc_auc_score(df.actual, df.positive_confidence),
                "true_negatives": cm[0][0],
                "false_positives": cm[0][1],
                "false_negatives": cm[1][0],
                "true_positives": cm[1][1],
                "mean_s": mean_s,
                "p50_s": p50_s,
                "p99_s": p99_s,
                "mean_classification_prompt_length": mean_classification_prompt_length,
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


def compare_runs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    y_col: str = "actual",
    yhat_col: str = "prediction",
    n_resamples: int = 10_000,
    alternative: str = "two-sided",
    statistic: str = "macro-f1",
) -> Tuple[float, float]:
    """
    Paired permutation test for the macro-F1 difference between two systems.

    Parameters
    ----------
    df_a, df_b   data-frames that contain the gold labels (`y_col`)
                 and system predictions (`yhat_col`).  They must refer
                 to **exactly the same set of articles** (order does not
                 matter - we align on index).
    y_col        column with ground-truth labels  (default: "actual")
    yhat_col     column with system predictions   (default: "prediction")
    n_resamples  number of permutation samples    (default: 10 000)
    alternative  "two-sided" | "greater" | "less" (default: "two-sided")

    Returns
    -------
    (delta, p)   delta  = F1_a - F1_b
                 p      = permutation p-value
    """

    if len(df_a) != len(df_b):
        raise ValueError("dataframes must cover the same articles")

    df_a = df_a.sort_values(by=["article_title"], ignore_index=True)
    df_b = df_b.sort_values(by=["article_title"], ignore_index=True)

    assert df_a.index.equals(df_b.index), "dataframes must cover the same articles"

    y_true = df_a[y_col].values
    yhat_a = df_a[yhat_col].values
    yhat_b = df_b[yhat_col].values

    # sanity check
    assert np.array_equal(y_true, df_b[y_col].values), "gold labels mismatch"

    # ------------------------------------------------------------------
    # 2  define the statistic (paired difference in macro-F1)
    # ------------------------------------------------------------------
    def stat(pred_a, pred_b):
        return f1_score(y_true, pred_a, average="macro") - f1_score(
            y_true, pred_b, average="macro"
        )

    delta_obs = stat(yhat_a, yhat_b)

    # ------------------------------------------------------------------
    # 3  permutation test (swap predictions article-wise)
    # ------------------------------------------------------------------
    res = permutation_test(
        data=(yhat_a, yhat_b),
        statistic=stat,
        permutation_type="pairings",  # swap within each article
        vectorized=False,
        n_resamples=n_resamples,
        alternative=alternative,
        random_state=0,
    )
    return float(delta_obs), float(res.pvalue)
