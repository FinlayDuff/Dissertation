# Core Folder Overview

The `core` folder contains the primary modules and classes that drive the misinformation detection system. This folder is the heart of the project, where the main logic and workflows are implemented. Below is a detailed overview of the key components within the `core` folder and how they interact with each other.

## Modules and Classes

### 1. Experiment

**File:** [experiment.py](experiment.py)

The `Experiment` class is responsible for managing misinformation detection experiments. It handles the configuration, execution, evaluation, and logging of experiments. The class integrates various components to create a cohesive workflow for detecting misinformation.

**Key Responsibilities:**
- **Configuration:** Initializes the experiment with a specific configuration.
- **Graph Execution:** Executes the workflow graph defined in the `GraphManager`.
- **Evaluation:** Uses the `evaluate` function from `langsmith` to assess the performance of the experiment.
- **Logging:** Logs the details and results of the experiment for future reference.

**Key Methods:**
- `__init__`: Initializes the experiment with a given configuration.
- `run_evaluation`: Runs the evaluation on a specified dataset.
- `log_experiment_details`: Logs the experiment details and results to a file.
- `_convert_results_to_dict`: Converts evaluation results to a serializable dictionary.
- `correct_label`: Corrects the label evaluator for LangSmith.

### 2. LLM Factory

**File:** [llm_factory.py](llm_factory.py)

The `LLMFactory` module is responsible for creating and managing instances of Large Language Models (LLMs). It provides a standardized way to initialize and configure different LLMs used in the misinformation detection system.

**Key Responsibilities:**
- **Model Initialization:** Initializes LLM instances based on the provided configuration.
- **Model Configuration:** Configures the models with necessary parameters such as model name, temperature, and timeout.

**Key Methods:**
- `create_model`: Creates and returns an instance of the specified LLM.

### 3. Misinformation Detection

**File:** [misinformation_detection.py](misinformation_detection.py)

The `MisinformationDetection` class is responsible for detecting misinformation in articles. It provides methods for classifying articles, detecting credibility signals, and running follow-up analyses.

**Key Responsibilities:**
- **Article Classification:** Classifies articles as credible or fake.
- **Signal Detection:** Detects credibility signals within the articles.
- **Follow-up Analysis:** Runs additional analyses based on the detected signals.

**Key Methods:**
- `classify_article`: Classifies an article as credible or fake.
- `detect_signals`: Detects credibility signals in the article.
- `critic_signal_classification`: Classifies the credibility signals.
- `run_followup_analysis`: Runs follow-up analyses on the detected signals.

### 4. Graph Manager

**File:** [graph_manager.py](graph_manager.py)

The `GraphManager` class manages the workflow graph for misinformation detection operations. It constructs and executes a directed graph of operations that route between different classification methods, handle credibility signal detection, and make decisions about additional analysis.

**Key Responsibilities:**
- **Graph Construction:** Builds the workflow graph with nodes and edges representing different operations.
- **Graph Execution:** Executes the graph on given examples to detect misinformation.
- **Conditional Branching:** Handles conditional branching between different operations based on the state.

**Key Methods:**
- `__init__`: Initializes the graph manager with a detection system and verbosity settings.
- `build_graph`: Constructs the workflow graph with nodes and edges.
- `decide_start_path`: Determines the initial path based on whether to use signals.
- `decide_critic_path`: Determines the path after the critic's decision.
- `decide_signals_critic_path`: Determines the path based on signal critiques.
- `run_graph_on_example`: Executes the graph on a given example.
- `visualize_graph`: Visualizes the constructed workflow graph.

## How They Work Together

1. **Experiment Initialization:**
   - The `Experiment` class is initialized with a specific configuration. It sets up the `MisinformationDetection` and `GraphManager` instances.

2. **Graph Construction:**
   - The `GraphManager` constructs the workflow graph using the methods provided by the `MisinformationDetection` class. Nodes and edges are added to represent different operations such as article classification and signal detection.

3. **Model Initialization:**
   - The `LLMFactory` is used to create and configure instances of LLMs required for the experiment. These models are used by the `MisinformationDetection` class for various tasks.

4. **Graph Execution:**
   - The `Experiment` class runs the evaluation by executing the workflow graph on the provided dataset. The `GraphManager` handles the execution, routing between different operations based on the state.

5. **Evaluation and Logging:**
   - The `Experiment` class uses the `evaluate` function to assess the performance of the experiment. It logs the details and results for future reference.

By integrating these components, the `core` folder provides a robust framework for conducting misinformation detection experiments. Each module plays a crucial role in ensuring the system's functionality, from initializing models to executing complex workflows and evaluating results.
