#!/bin/bash

# Enable error handling
set -e

# Check if yq is installed (tool for parsing YAML)
if ! command -v yq &> /dev/null; then
    echo "yq is required but not installed. Installing..."
    brew install yq
fi

# Set cache directory
CACHE_DIR="${HOME}/.cache/huggingface/hub"

# Read models from YAML
CONFIG_FILE="$(dirname $0)/../config/hugging_face_models.yml"
echo "Reading config from $CONFIG_FILE"

# Set Hugging Face Hub to online mode
export HF_HUB_OFFLINE=0

# Loop through models in YAML
for model in $(yq e '.models | keys | .[]' "$CONFIG_FILE"); do
    model_id=$(yq e ".models.$model.name" "$CONFIG_FILE")
    
    model_namespace=${model_id%%/*}
    model_name=${model_id##*/}
    model_cache_path="$CACHE_DIR/models--${model_namespace}--${model_name}"

    if [ -d "$model_cache_path" ]; then
        echo "✓ Model $model ($model_id) already exists in cache, skipping..."
        continue
    fi
    
    echo "Downloading $model model ($model_id)..."
    transformers-cli download --cache-dir "$CACHE_DIR" "$model_id"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded $model"
    else
        echo "✗ Failed to download $model"
        exit 1
    fi
done

echo "All models downloaded successfully!"

# Set Hugging Face Hub to offline mode
export HF_HUB_OFFLINE=1