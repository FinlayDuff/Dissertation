install:
	poetry install

transform_raw_datasets:
	poetry run python -m scripts.dataset_prep

upload_langsmith_dataset:
	poetry run python -m scripts.langsmith_dataset_upload

upload_langsmith_dataset_clean:
	poetry run python -m scripts.langsmith_dataset_upload --overwrite

setup_datasets_clean: transform_raw_datasets upload_langsmith_dataset_clean

add_new_datasets: transform_raw_datasets upload_langsmith_dataset