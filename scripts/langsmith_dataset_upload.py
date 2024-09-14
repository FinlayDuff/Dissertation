import os
from utils.data.csv_parsing import load_csv_as_dataframe
from utils.data.langsmith_dataset import (
    upload_langsmith_dataset,
    get_current_datasets,
    delete_dataset,
)
from utils.utils import load_config


# Function to load a dataset based on the config
def upload_dataset(dataset_info):

    name = dataset_info["name"]
    path = dataset_info["path"]

    dataset_path = os.getcwd() + path

    # Load the dataset
    print(f"Loading dataset: {name}")
    df = load_csv_as_dataframe(dataset_path)

    # Uploading dataset to Langsmith
    print(f"Uploading dataset: {name}")
    upload_langsmith_dataset(
        df=df,
        input_keys=dataset_info["input_keys"],
        output_keys=dataset_info["output_keys"],
        name=name,
        description=dataset_info["description"],
    )


# Function to load all datasets defined in the config
def upload_datasets(config_file):
    config = load_config(config_file)
    current_datasets_dict = get_current_datasets()
    for dataset_info in config["datasets"]:
        if dataset_info["name"] in current_datasets_dict.keys():
            print(f"Deleting dataset {dataset_info['name']} from LangSmith")
            delete_dataset(dataset_info["name"])
        upload_dataset(dataset_info)


# Main script to run the loading process
if __name__ == "__main__":
    config_path = "config/datasets.yml"
    datasets = upload_datasets(config_path)
