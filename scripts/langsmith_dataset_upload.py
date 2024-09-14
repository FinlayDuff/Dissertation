import os
from utils.data.csv_parsing import load_csv_as_dataframe
from utils.data.langsmith_dataset import LangsmithDatasetManager
from utils.utils import load_config


# Function to load a dataset based on the config
def upload_dataset(dataset_info, dataset_manager):

    name = dataset_info["name"]
    path = dataset_info["path"]

    dataset_path = os.getcwd() + path

    # Load the dataset
    print(f"Loading dataset {name} into memory")
    df = load_csv_as_dataframe(dataset_path)

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
def upload_datasets(config_file, dataset_manager):
    config = load_config(config_file)
    current_datasets_dict = dataset_manager.get_current_datasets()
    for dataset_info in config["datasets"]:
        if dataset_info["name"] in current_datasets_dict.keys():
            print(f"Deleting dataset {dataset_info['name']} from LangSmith")
            dataset_manager.delete_dataset(dataset_info["name"])
        upload_dataset(dataset_info, dataset_manager)


# Main script to run the loading process
if __name__ == "__main__":
    config_path = "config/datasets.yml"
    manager = LangsmithDatasetManager()  # Instantiating the manager
    upload_datasets(config_path, manager)
