# Finlay Duff Dissertation - Bath Masters in Artificial Intelligence 2024

## Overview
This repository represents the code of the research done for my masters' dissertation.

You can find my paper here: 

In my dissertation, I investigate a model-in-the-loop approach utilising Large Language Models (LLMs) to combat the pervasive problem of fake news. Leveraging the advanced capabilities of LLM-agents, this project aims to enhance the detection and verification of misinformation. By incorporating the latest research on fake news characteristics and misinformation classification methods, this proposal outlines a multi-faceted approach that combines the deep learning capabilities of NLP techniques with the dynamic interactive potential of LLMs. This integrated system will not only identify deceptive content but also provide human-like reasoned explanations for its analysis. 

## Installation Instructions
Poetry is used for package management with a python version of ^3.10. Ensure that your environment is pointing at a version of python compatible with the Poetry project. 

__Requirements__
You will require:
1. A valid LANGSMITH_API_KEY added to your environment
2. A valid OPENAI_API_KEY or equivalent added to your environment

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


__Run Experiment__
You can run an pre-defined experiment on an uploaded dataset by running this: 
```bash
python main.py  --experiment=few_shot_gpt4  --dataset="isot_2000"
```
Refer to config/experiments.py and config/datasets.yml for possible combinations.

## Contents
```bash
├── README.md                 # Project overview and instructions
├── pyproject.toml            # Python project configuration and dependencies
├── poetry.lock               # Locked Python dependencies
├── Makefile                  # Build and management commands
├── main.py                   # Main entry point of the application
├── docker-compose.yml        # Docker orchestration configuration
├── run_experiment.sh         # Script to execute experiments
├── .env                      # Environment variables
├── config                    # Configuration files
│   ├── datasets.yml          # Dataset configurations
│   ├── experiments.py        # Experiment configurations
│   ├── prompts.py            # Prompt configurations for LLMs
│   ├── signals.py            # Signal configurations
│   └── hugging_face_models.yml  # Hugging Face model configurations
├── core                      # Core modules and classes
│   ├── __init__.py
│   ├── experiment.py         # Experiment management and evaluation
│   ├── graph_manager.py      # Workflow graph management
│   ├── llm_factory.py        # LLM initialization and configuration
│   ├── misinformation_detection.py  # Misinformation detection logic
│   ├── followup_analysis_tools.py     # Analysis tools for follow-up tasks
│   ├── state.py              # State management across modules
│   └── README.md             # Core folder overview
├── data                      # Data storage
│   ├── raw                   # Raw datasets
│   ├── transformed           # Transformed datasets for analysis
│   └── README.md             # Data directory overview
├── logs                      # Log files from various processes
├── notebooks                 # Jupyter notebooks for experiments and analysis
│   ├── __init__.py
│   └── zero_shot.ipynb       # Zero-shot learning experiments
├── results                   # Experiment result outputs
├── scripts                   # Utility scripts for auxiliary tasks
│   └── __init__.py
└── utils                     # Helper functions and utilities
```