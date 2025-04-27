from utils.logging import setup_logging
from langchain.globals import set_debug
import argparse
import multiprocessing


def main():
    parser = argparse.ArgumentParser(
        description="Run misinformation detection experiment"
    )
    parser.add_argument("--experiment", default="baseline_claude_3_5_haiku")
    parser.add_argument("--dataset", default="FA-KES debug")
    parser.add_argument("--chunk", action="store_true")
    parser.add_argument("--evaluate_locally", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_debug(False)

    logger = setup_logging(
        verbose=args.verbose, app_name=f"{args.dataset}_{args.experiment}"
    )

    # Import experiment due to logging setup
    from core.experiment import Experiment

    experiment = Experiment(
        experiment_name=args.experiment,
        dataset_name=args.dataset,
        description="Evaluating graph-based misinformation detection system.",
        chunk=args.chunk,
        evaluate_locally=args.evaluate_locally,
        verbose=args.verbose,
    )

    experiment.run()
    logger.info(f"Experiment {args.dataset}_{args.experiment} completed successfully.")
    logger.info("Results saved to results directory.")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
