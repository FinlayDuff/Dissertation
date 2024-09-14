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

__Install dependencies:__
```bash
poetry install
```

## Contents
```bash
├── README.md                 # Overview of the project
├── pyproject.toml            # Dependencies
├── data
│   ├── raw                   # Raw datasets
│   ├── processed             # Preprocessed datasets
│   └── splits                # Train, validation, and test splits
├── notebooks
│   ├── exploration           # Data exploration and initial analysis
│   ├── baseline              # Baseline model experiments
│   └── complex_models        # Advanced model experiments
├── scripts
│   ├── data_preprocessing.py # Data cleaning and preprocessing
│   ├── train_baseline.py     # Training baseline models
│   ├── evaluate.py           # Evaluating models
│   └── train_complex.py      # Training complex models
├── models
│   ├── baseline              # Saved baseline models
│   └── complex               # Saved advanced models
├── results
│   ├── baseline              # Results from baseline models
│   └── complex               # Results from complex models
├── logs
│   ├── training_logs         # Logs from model training sessions
│   └── evaluation_logs       # Logs from model evaluations
└── config
    └── config.yaml           # Configuration files for experiments
```