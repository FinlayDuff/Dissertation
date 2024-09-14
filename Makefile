install:
	poetry install

transform_raw_datasets:
	poetry run python -m scripts.dataset_prep

upload_langsmith_dataset:
	poetry run python -m scripts.langsmith_dataset_upload
