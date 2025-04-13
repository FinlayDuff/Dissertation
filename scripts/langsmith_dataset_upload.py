import os
from utils.data.csv_parsing import load_csv_as_dataframe
from utils.data.langsmith_dataset import LangsmithDatasetManager, get_manager
from utils.utils import load_config
import pandas as pd
import argparse

from utils.data import TRANSFORM_FUNCTIONS


def upload_dataset(dataset_info: dict, dataset_manager: LangsmithDatasetManager):

    name = dataset_info["name"]
    path = dataset_info["path"]
    transform_key = dataset_info.get("transform")
    total_samples = dataset_info.get("total_samples", None)

    dataset_path = os.getcwd() + path

    # Apply transformation if specified
    if transform_key and transform_key in TRANSFORM_FUNCTIONS:
        transform_func = TRANSFORM_FUNCTIONS[transform_key]
        print(f"Transforming dataset '{name}' with total_samples={total_samples}")
        transformed_file_path = transform_func(
            dataset_path=dataset_path,
            total_samples=total_samples,
        )
    else:
        print(f"No transform function specified. Using raw path for dataset '{name}'.")

    # Load transformed CSV (will have few_shot column if requested)
    print(f"Loading dataset {name} into memory")
    df = load_csv_as_dataframe(transformed_file_path)
    df = df[dataset_info["input_keys"] + dataset_info["output_keys"]]
    size_mb = df.memory_usage(deep=True).sum() / 1e6
    if size_mb > 20:
        raise RuntimeError(
            f"Dataset too large ({size_mb:.2f}MB). LangSmith max is 20MB."
        )

    print(f"Uploading dataset {name} into LangSmith")
    dataset_manager.upload_dataset(
        df=df,
        input_keys=dataset_info["input_keys"],
        output_keys=dataset_info["output_keys"],
        name=name,
        description=dataset_info["description"],
    )

    chunk_size = dataset_info.get("chunk_size", None)
    if chunk_size:

        dataset_manager.split_dataset_manual(
            df=df,
            dataset_name=name,
            input_keys=dataset_info["input_keys"],
            output_keys=dataset_info["output_keys"],
            chunk_size=chunk_size,
            description=dataset_info["description"],
        )


# Function to load all datasets defined in the config
def upload_datasets(config_file, dataset_manager, overwrite=False):
    config = load_config(config_file)
    current_datasets_dict = dataset_manager.get_current_datasets()
    for dataset_info in config["datasets"]:
        base_name = dataset_info["name"]
        if base_name in current_datasets_dict.keys():
            if overwrite:
                print(f"Deleting dataset {dataset_info['name']} from LangSmith")
                dataset_manager.delete_dataset(base_name)
                # Delete chunked datasets
                chunked_names = [
                    name
                    for name in current_datasets_dict
                    if name.startswith(f"{base_name} [chunk")
                ]
                for chunk_name in chunked_names:
                    print(f"[INFO] Deleting chunked dataset: {chunk_name}")
                    dataset_manager.delete_dataset(chunk_name)
                upload_dataset(dataset_info, dataset_manager)
            else:
                print("Dataset already exists and overwrite == False. Skipping upload.")
        else:
            upload_dataset(dataset_info, dataset_manager)


# Main script to run the loading process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run misinformation detection experiment"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="OVERWRITE existing datasets"
    )
    args = parser.parse_args()

    config_path = "config/datasets.yml"
    manager = get_manager()  # Instantiating the manager
    upload_datasets(config_path, manager, args.overwrite)
