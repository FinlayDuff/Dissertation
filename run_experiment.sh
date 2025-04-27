#!/bin/bash

# Enable error handling
set -e

# Cleanup function
cleanup() {
    echo "Cleaning up containers..."
    docker-compose down
    exit 0
}

# Set up trap for script interruption
trap cleanup SIGINT SIGTERM

# Check if experiment name is provided
if [ -z "$1" ]
then
    echo "Please provide an experiment name"
    echo "Usage: ./run_experiment.sh <experiment_name> [dataset_name]"
    exit 1
fi

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker-compose down

# Export variables for docker-compose
export EXPERIMENT=$1
export DATASET=$2

# Set dataset name or use default
DATASET=${2:-FA-KES}
export DATASET

# Show configuration
echo "Running experiment with configuration:"
echo "- Experiment: $EXPERIMENT"
echo "- Dataset: $DATASET"
echo

# Run the experiment
echo "Building container..."
if ! docker-compose build; then
    echo "Build failed!"
    cleanup
    exit 1
fi

echo "Running experiment..."
if ! docker-compose run misinformation_detector; then
    echo "Experiment failed with exit code $?"
    cleanup
    exit 1
fi

# Successful completion
echo "Experiment completed successfully"
cleanup