import os
from utils.data.csv_parsing import load_csv_as_dataframe
from utils.data.langsmith_dataset import LangsmithDatasetManager
from utils.utils import load_config
import pandas as pd
import argparse


def balance_dataset_to_row_count(df, label_column, row_count):
    # Calculate rows per label
    unique_labels = df[label_column].nunique()
    rows_per_label = row_count // unique_labels

    # Sample rows_per_label rows from each label
    balanced_df = (
        df.groupby(label_column)
        .apply(lambda x: x.sample(n=rows_per_label, random_state=42))
        .reset_index(drop=True)
    )

    # Handle cases where exact row_count isn't possible due to uneven division
    if len(balanced_df) < row_count:
        # Sample additional rows to meet row_count
        extra_needed = row_count - len(balanced_df)
        extra_samples = df.sample(n=extra_needed, random_state=42)
        balanced_df = pd.concat([balanced_df, extra_samples]).reset_index(drop=True)

    return balanced_df


# Function to load a dataset based on the config
def upload_dataset(dataset_info, dataset_manager):

    name = dataset_info["name"]
    path = dataset_info["path"]
    row_count = dataset_info.get("row_count")

    dataset_path = os.getcwd() + path

    # Load the dataset
    print(f"Loading dataset {name} into memory")
    df = load_csv_as_dataframe(dataset_path)

    if row_count:
        # Balance the dataset to the desired row count
        print(f"Balancing dataset {name} to {row_count} rows")
        df = balance_dataset_to_row_count(df, "label", row_count)

    # Uploading dataset to Langsmith
    print(f"Uploading dataset {name} into LangSmith")
    dataset_manager.upload_dataset(
        df=df,
        input_keys=dataset_info["input_keys"],
        output_keys=dataset_info["output_keys"],
        name=name,
        description=dataset_info["description"],
    )


# Function to load all datasets defined in the config
def upload_datasets(config_file, dataset_manager, overwrite=False):
    config = load_config(config_file)
    current_datasets_dict = dataset_manager.get_current_datasets()
    for dataset_info in config["datasets"]:
        if dataset_info["name"] in current_datasets_dict.keys():
            if overwrite:
                print(f"Deleting dataset {dataset_info['name']} from LangSmith")
                dataset_manager.delete_dataset(dataset_info["name"])
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
    manager = LangsmithDatasetManager()  # Instantiating the manager
    upload_datasets(config_path, manager, args.overwrite)
