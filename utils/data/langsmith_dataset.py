from langsmith import Client
from pandas import DataFrame
from typing import List

client = Client()


def upload_langsmith_dataset(
    df: DataFrame, input_keys: List, output_keys: list, name: str, description: str
):

    client.upload_dataframe(
        df=df,
        input_keys=input_keys,
        output_keys=output_keys,
        name=name,
        description=description,
        data_type="kv",
    )


def get_current_datasets():
    current_datasets = client.list_datasets()
    return {datasets.name: datasets for datasets in current_datasets}


def delete_dataset(dataset_name):
    client.delete_dataset(dataset_name=dataset_name)
