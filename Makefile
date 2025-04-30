.PHONY: install setup_spacy upload_langsmith_dataset upload_langsmith_dataset_clean download_hugging_face_models

setup: install setup_spacy upload_langsmith_dataset_clean download_hugging_face_models

install:
	echo "Installing dependencies. This may take a while..."
	poetry install

setup_spacy:
	poetry run python -m spacy download en_core_web_sm

upload_langsmith_dataset:
	poetry run python -m scripts.langsmith_dataset_upload

upload_langsmith_dataset_clean:
	echo "Transforming datasets and uploading to Langsmith. This may take a while..."
	poetry run python -m scripts.langsmith_dataset_upload --overwrite

download_hugging_face_models:
	echo "Downloading Hugging Face models. This may take a while..."
	./scripts/hugging_face_download.sh
