# main.py
import argparse
from core.experiment import Experiment
from langchain.globals import set_debug


def main(
    experiment_name: str = "baseline_gpt4",
    dataset_name: str = "FA-KES debug",
    verbose: bool = False,
):
    """
    Run a misinformation detection experiment.

    Args:
        experiment_name: Name of experiment config to use
        dataset_name: Name of dataset to evaluate
        verbose: Whether to print debug information
    """

    set_debug(False)

    # Initialize and run experiment
    experiment = Experiment(
        experiment_name=experiment_name,
        description="Evaluating graph-based misinformation detection system.",
        verbose=verbose,
    )

    # Run evaluation
    experiment.run_evaluation(dataset_name)
    if verbose:
        print(f"Experiment {experiment_name} completed.")
        print(f"Results saved to results directory.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run misinformation detection experiment"
    )
    parser.add_argument(
        "--experiment",
        default="baseline_claude_3_5_haiku",
        help="Name of experiment configuration to use",
    )
    parser.add_argument(
        "--dataset", default="FA-KES debug", help="Name of dataset to evaluate"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debug information"
    )

    args = parser.parse_args()
    main(args.experiment, args.dataset, args.verbose)
