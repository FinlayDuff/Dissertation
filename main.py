from core.experiment import Experiment
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

    experiment = Experiment(
        experiment_name=args.experiment,
        dataset_name=args.dataset,
        description="Evaluating graph-based misinformation detection system.",
        chunk=args.chunk,
        evaluate_locally=args.evaluate_locally,
        verbose=args.verbose,
    )
    experiment.run()
    if args.verbose:
        print(f"Experiment {args.experiment} completed.")
        print("Results saved to results directory.")


if __name__ == "__main__":
    # Optional: safer multiprocessing backend on macOS
    multiprocessing.set_start_method("spawn", force=True)
    main()
