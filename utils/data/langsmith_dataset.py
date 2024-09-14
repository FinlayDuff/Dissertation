from langsmith import Client
from pandas import DataFrame
from typing import List


class LangsmithDatasetManager:
    def __init__(self, client=None):
        self.client = client or Client()

    def upload_dataset(
        self,
        df: DataFrame,
        input_keys: List,
        output_keys: list,
        name: str,
        description: str,
    ):
        self.client.upload_dataframe(
            df=df,
            input_keys=input_keys,
            output_keys=output_keys,
            name=name,
            description=description,
            data_type="kv",
        )

    def get_current_datasets(self):
        current_datasets = self.client.list_datasets()
        return {datasets.name: datasets for datasets in current_datasets}

    def delete_dataset(self, dataset_name: str):
        self.client.delete_dataset(dataset_name=dataset_name)
