# Technical Implementation Documentation

This document provides an in-depth explanation of the technical implementation of the misinformation detection system. It covers the core modules, their interactions, configuration details, and the workflow orchestration designed to leverage Large Language Models (LLMs) for detecting fake news. This document is intended to serve as a comprehensive guide for developers, researchers, and contributors.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Components](#core-components)
   - [Experiment](#experiment)
   - [LLM Factory](#llm-factory)
   - [Misinformation Detection](#misinformation-detection)
   - [Graph Manager](#graph-manager)
3. [Workflow and Data Flow](#workflow-and-data-flow)
4. [Configuration Management](#configuration-management)
5. [LLM Integration](#llm-integration)
6. [Parsing and Response Handling](#parsing-and-response-handling)
7. [Development and Deployment](#development-and-deployment)
8. [Appendix: References](#appendix-references)

---

## Project Overview

The misinformation detection system is built to leverage state-of-the-art LLMs for classifying articles as credible or fake, identifying credibility signals, and performing follow-up analyses. The project integrates multiple components to create a dynamic, model-in-the-loop approach. The key objectives are to:

- **Initialize and configure** various LLMs dynamically.
- **Construct and manage** a workflow graph that defines the sequence of operations.
- **Facilitate end-to-end experiments** for misinformation analysis.
- **Integrate modular parsing routines** to handle LLM responses.

The project structure is organized as follows:

- **Core Modules:** Located in the `core/` directory ([llm_factory.py](core/llm_factory.py), [experiment.py](core/experiment.py), [graph_manager.py](core/graph_manager.py), and [misinformation_detection.py](core/misinformation_detection.py)).
- **Configurations:** Stored in the `config/` directory ([experiments.py](config/experiments.py), [prompts.py](config/prompts.py), [signals.py](config/signals.py)).
- **Utilities:** Helper functions and scripts reside in the `utils/` folder.
- **Notebooks and Scripts:** Jupyter notebooks to run experiments and test functionalities are in `notebooks/`.

---

## Core Components

### Experiment

The `Experiment` class ([experiment.py](core/experiment.py)) is the orchestrator of the entire misinformation detection process. Its responsibilities include:

- **Configuration:** It parses experimental configurations defined in the `config/experiments.py` file.
- **Graph Execution:** It integrates with the [`GraphManager`](core/graph_manager.py) to build an execution graph.
- **Evaluation:** It utilizes both local and remote evaluation libraries (e.g., LangSmith) to assess model performance.
- **Logging:** Detailed logging of experiments is provided for diagnostics and future reference.

*Key Technical Aspects:*
- Uses multiprocessing (via `cpu_count`) to optimize evaluations.
- Integrates with external libraries like `tqdm` for progress tracking.
- Adheres to a strict data flow which is transformed between various formats (JSON, Python dicts).

### LLM Factory

The `LLMFactory` module ([llm_factory.py](core/llm_factory.py)) centralizes the initialization and configuration of LLMs. Its key functions include:

- **Dynamic Model Selection:** Based on the experiment configuration and state, the appropriate LLM is selected using [`get_llm_from_model_name`](utils/langchain/llm_model_selector.py).
- **Configuration Wrapping:** It wraps configurations in the [`LLMConfig`](core/llm_factory.py#L65) dataclass which includes task-specific prompts, model parameters (temperature, timeout, etc.), and state-driven customizations.
- **Parsing Mechanism:** Offers methods for parsing LLM responses (e.g., `_parse_classification`, `_parse_bulk_signals`). These methods inspect the JSON output from the LLM and ensure that key criteria (such as required fields) are met.

*Key Technical Aspects:*
- Utilizes Python’s `dataclass` for configuration encapsulation.
- Implements static methods to create specialized configurations depending on task type (zero-shot, few-shot, bulk signals, etc.).
- Custom parsing methods use regex to strip code fences and validate expected JSON structure.

### Misinformation Detection

The [`MisinformationDetection`](core/misinformation_detection.py) class encapsulates the logic to detect fake news. It interacts closely with the LLM instances initialized by the `LLMFactory` to:

- **Article Classification:** Determine if an article is credible or fake.
- **Signal Detection:** Identify various credibility signals embedded in the article.
- **Follow-up Analysis:** Trigger detailed analyses based on initial classifications and additional signals.

*Key Technical Aspects:*
- Integrates with the Experiment and Graph Manager modules to provide a seamless workflow.
- Uses state management to pass data between different stages of the detection process.
- Provides hooks for both individual and bulk analysis paradigms.

### Graph Manager

The `GraphManager` ([graph_manager.py](core/graph_manager.py)) is responsible for constructing and executing a directed workflow graph. It supports:

- **Graph Construction:** Nodes represent tasks such as article classification or signal detection, dynamically created based on the input state.
- **Conditional Branching:** Based on LLM outputs and state, the graph manager decides the subsequent path (e.g., whether to run follow-up analysis).
- **Visualization:** The graph can be visualized to aid debugging and understanding of the overall execution flow.

*Key Technical Aspects:*
- Builds graphs implementing conditional logic.
- Integrates with both the Experiment and LLMFactory for a runtime decision-making process.
- Provides a clear separation of concerns between graph construction and execution.

---

## Workflow and Data Flow

1. **Experiment Initialization:**
   - The `Experiment` object is created with a configuration dict loaded from [`config/experiments.py`](config/experiments.py).
   - It instantiates required core modules including the [`MisinformationDetection`](core/misinformation_detection.py) and [`GraphManager`](core/graph_manager.py).

2. **LLM Creation and Configuration:**
   - Depending on the type of task (e.g., zero-shot classification vs. bulk signals), the `LLMFactory` creates a configured LLM.
   - Each LLM is wrapped with custom parsing logic. For example, `_parse_bulk_signals` in [`llm_factory.py`](core/llm_factory.py) parses structured outputs to ensure consistency.

3. **Graph Execution:**
   - The workflow graph defined in the graph manager is executed.
   - Nodes interact by passing a `state` (managed by [`State`](core/state.py)) that carries article content, signal configurations, and responses.
   - Conditional logic (e.g., checking if `few_shot` examples exist) governs the graph’s branching.

4. **Evaluation and Logging:**
   - Results from the LLM are parsed and logged. The experiment module logs evaluation metrics using remote evaluation functions (from `langsmith.evaluation`) as well as local custom evaluators.
   - Detailed logs are stored in the `logs/` directory to track model performance over iterations.

---

## Configuration Management

The system relies heavily on configuration defined in the following files:

- **Experiments:** [`config/experiments.py`](config/experiments.py) maintains different experimental setups including model parameters and toggles such as `few_shot` or `use_bulk_signals`.
- **Prompts:** [`config/prompts.py`](config/prompts.py) provides the various prompts used by the LLMs. These include instructions for classification, signal detection, and critic operations.
- **Signals:** [`config/signals.py`](config/signals.py) creates a mapping for credibility signals. Both condensed and standard signal sets are defined here, influencing how the LLM parses output signals.

The configurations are passed along through the `LLMConfig` dataclass in [`llm_factory.py`](core/llm_factory.py), ensuring consistency and ease of debugging.

---

## LLM Integration

A key feature of the system is the integration with various Large Language Models (LLMs). The approach involves:

- **Dynamic LLM Selection:** The function [`get_llm_from_model_name`](utils/langchain/llm_model_selector.py) determines the model to initialize based on the experiment configuration.
- **Custom Response Wrappers:** LLM responses are wrapped in a subclass (`WrappedLLM`) that overrides the `invoke()` method. This allows a uniform way to perform:
  - **Message Formatting:** Combining system prompts with user content.
  - **Response Parsing:** Depending on the task type (`TaskType` enum), specific parsing functions are invoked:
    - `_parse_classification` for article classification.
    - `_parse_bulk_signals` for bulk credibility signal parsing.
    - `_parse_signal` and `_parse_critic` for individual tasks.
- **Error Handling:** Robust error handling is integrated to capture failures in JSON decoding or missing keys in the output.

---

## Parsing and Response Handling

Parsing routines are crucial for converting LLM output into meaningful structured data. The parsing mechanism includes:

- **Regex Based Cleanup:** The parsers use regular expressions to strip markdown code fences (e.g., ```json) to isolate pure JSON content.
- **Validation Checks:** After JSON parsing, the code verifies that necessary keys (e.g., `label`, `confidence`, `explanation`) are present.
- **State-based Decisions:** For bulk signal responses, the parser `_parse_bulk_signals` selects either the condensed or full signal questions based on the `state` flag.

These methods ensure that downstream components receive well-interpreted data for further processing.

---

## Development and Deployment

### Development

- **Environment Management:** The project uses Poetry for dependency management. Ensure that your Python version is compatible with `pyproject.toml` specifications.
- **Testing:** Unit tests and integration tests should be added to validate individual components (e.g., LLM response parsing, graph execution flow).
- **Notebooks:** The `notebooks/` directory ([zero_shot.ipynb](notebooks/zero_shot.ipynb)) contains experimental code for quick prototyping and demonstration.

### Deployment

- **Dockerization:** The project includes a `Dockerfile` and `docker-compose.yml` file for containerized execution.
- **Scripts:** Utility scripts (e.g., `run_experiment.sh`) are provided for running experiments in both local and production environments.
- **Logging:** Detailed logs are maintained in the `logs/` directory to monitor runtime performance and for post-mortem analysis.

---

## Appendix: References

- **Experiment Module:** [experiment.py](core/experiment.py)
- **LLM Factory Module:** [llm_factory.py](core/llm_factory.py)
- **Graph Manager Module:** [graph_manager.py](core/graph_manager.py)
- **Misinformation Detection:** [misinformation_detection.py](core/misinformation_detection.py)
- **Configuration Files:** [experiments.py](config/experiments.py), [prompts.py](config/prompts.py), [signals.py](config/signals.py)
- **LLM Selector:** [llm_model_selector.py](utils/langchain/llm_model_selector.py)

---
---

By integrating these components, the system achieves a robust and configurable platform for misinformation detection, leveraging modern LLMs along with a flexible workflow management infrastructure. The detailed documentation here should serve as both a reference and a guide for anyone looking to understand or extend the system.


# The Graph
The graph is essentially the backbone of the workflow—it orchestrates the order and conditional execution of all the tasks in the misinformation detection pipeline. Here's what it does:

Node Representation:
Each node in the graph represents a specific task (for example, article classification, signal detection, follow-up analysis, etc.). Every task is encapsulated as an independent module or function.

Conditional Branching:
The graph isn’t a simple linear sequence. It incorporates conditional logic so that the execution path can change depending on the output of earlier tasks. For example, if the initial classification indicates low credibility, the graph can branch to trigger more in-depth follow-up analysis.

State Passing:
At each step, nodes pass along and update a shared state (which holds the article content, signal configurations, results, etc.). This state is then used by subsequent nodes to make context-aware decisions.

Graph Execution:
The Graph Manager builds this directed graph from the provided configuration and then executes it in order—ensuring that each task is run, its output is parsed, and any conditional logic is applied to determine the next node to run.

Separation of Concerns:
The graph structure abstracts the order of execution from the individual task implementations. This means that you can modify, add, or remove tasks without affecting the inner logic of each module.

Visualization and Debugging:
The system includes the capability to visualize the graph, which helps developers understand the current workflow, debug issues, or optimize the process.

In short, the graph takes care of sequencing, branching, and data management, making sure that the right operations are performed at the right time based on both preconfigured rules and dynamic outcomes as the misinformation detection process unfolds.