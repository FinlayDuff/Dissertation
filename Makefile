install:
	poetry install

upload_langsmith_dataset:
	poetry run python -m scripts.langsmith_dataset_upload

upload_langsmith_dataset_clean:
	poetry run python -m scripts.langsmith_dataset_upload --overwrite
