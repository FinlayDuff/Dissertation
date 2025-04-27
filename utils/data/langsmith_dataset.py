from langsmith import Client
from pandas import DataFrame
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


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

    def split_dataset_manual(
        self,
        df: DataFrame,
        dataset_name: str,
        input_keys: list,
        output_keys: list,
        chunk_size: int,
        description: str,
    ):
        """
        Manually splits a dataframe into multiple datasets (chunk 1, chunk 2, ...) and uploads each.
        """
        total_rows = len(df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            chunk_name = f"{dataset_name} [chunk {i + 1}]"

            print(f"Uploading split: {chunk_name} with {len(chunk_df)} rows")
            self.upload_dataset(
                df=chunk_df,
                input_keys=input_keys,
                output_keys=output_keys,
                name=chunk_name,
                description=f"{description} (split {i+1}/{num_chunks})",
            )

    def get_dataset_chunks(self, dataset_name) -> List[tuple[int, str]]:
        """
        Retrieve all chunked versions of the dataset.

        Returns:
            List of tuples containing (chunk_number, dataset_name)
            e.g., [(1, 'FA-KES [chunk 1]'), (2, 'FA-KES [chunk 2]'), ...]

        Raises:
            ValueError: If no chunks are found for the given dataset name
            RuntimeError: If failed to retrieve dataset chunks
        """
        try:
            # Get all datasets first
            all_datasets = self.client.list_datasets()

            # Extract chunk numbers and create tuples of (name, id)
            chunked_datasets_with_id = []
            for ds in all_datasets:
                try:
                    if ds.name.startswith(f"{dataset_name} [chunk "):
                        chunk_id = int(ds.name.split("chunk ")[-1].strip("]"))
                        chunked_datasets_with_id.append((chunk_id, ds.name))
                except (ValueError, IndexError):
                    continue

            if not chunked_datasets_with_id:
                raise ValueError(f"No dataset chunks found for {dataset_name}")

            # Sort based on chunk number
            return sorted(chunked_datasets_with_id, key=lambda x: x[0])
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve dataset chunks: {str(e)}")

    def load_dataset(self, dataset_name: str):
        """
        Load examples from a LangSmith dataset.

        Args:
            dataset_name: Name of the dataset in LangSmith

        Returns:
            List of langsmith.schemas.Example
        """
        return list(self.client.list_examples(dataset_name=dataset_name))


_instance: Optional[LangsmithDatasetManager] = None


def get_manager() -> LangsmithDatasetManager:
    global _instance
    if _instance is None:
        logger.info("Initialising the Langsmith Dataset Manager")
        _instance = LangsmithDatasetManager()
    return _instance
