# Finlay Duff Dissertation - Bath Masters in Artificial Intelligence 2024

## Overview
This repository represents the code of the research done for my masters' dissertation.

You can find my paper here: 

In my dissertation, I investigate a model-in-the-loop approach utilising Large Language Models (LLMs) to combat the pervasive problem of fake news. Leveraging the advanced capabilities of LLM-agents, this project aims to enhance the detection and verification of misinformation. By incorporating the latest research on fake news characteristics and misinformation classification methods, this proposal outlines a multi-faceted approach that combines the deep learning capabilities of NLP techniques with the dynamic interactive potential of LLMs. This integrated system will not only identify deceptive content but also provide human-like reasoned explanations for its analysis. 

## Installation Instructions
Poetry is used for package management with a python version of ^3.10. Ensure that your environment is pointing at a version of python compatible with the Poetry project. 

__Install poetry:__
```bash
pip install poetry
```

__(Option 1) Quick-start__:
To simplify installation, simply run ```make setup```.
This will perform the following:
1. Install the python packages and related dependencies
2. Transform and upload the datasets to Langsmith
3. Download all the hugging_face models

__(Option 2) Run the make files separately:__
e.g ```make install```



## Contents
```bash
├── README.md                 # Overview of the project
├── pyproject.toml            # Dependencies
├── poetry.lock               # Locked dependencies
├── Makefile                  # Build and management commands
├── main.py                   # Main entry point of the application
├── config                    # Configuration files
│   ├── datasets.yml          # Dataset configurations
│   ├── experiments.py        # Experiment configurations
│   ├── prompts.py            # Prompt configurations
│   └── signals.py            # Signal configurations
├── core                      # Core modules and classes
│   ├── __init__.py
│   ├── experiment.py         # Experiment management
│   ├── graph_manager.py      # Workflow graph management
│   ├── llm_factory.py        # LLM initialization and configuration
│   ├── misinformation_detection.py # Misinformation detection logic
│   ├── state.py              # State management
│   └── README.md             # Core folder overview
├── data                      # Data storage
│   ├── raw                   # Raw datasets
│   ├── transformed           # Transformed datasets
│   └── README.md             # Data folder overview
├── logs                      # Logs from various processes
├── models                    # Saved models
├── notebooks                 # Jupyter notebooks
│   ├── __init__.py
│   └── zero_shot.ipynb       # Zero-shot learning experiments
├── results                   # Experiment results
├── scripts                   # Utility scripts
│   ├── __init__.py
├── utils                     # Utility functions and helpers
```