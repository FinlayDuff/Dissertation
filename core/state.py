# state.py
from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Annotated


class State(TypedDict):
    """
    Complete state object tracking workflow progress and results.
    """

    # Input data
    article_title: Annotated[str, "Title of the article to analyze"]
    article_content: Annotated[str, "Content of the article to analyze"]

    # Configuration
    experiment_name: Annotated[str, "Name of the experiment configuration to use"]
    use_signals: Annotated[bool, "Whether to use credibility signal detection"]
    use_bulk_signals: Annotated[
        bool, "Whether to detect signals in bulk or individually"
    ]
    condensed_signals: Annotated[
        "bool", "Whether to use the condensed set of signals or the full list"
    ]
    few_shot: Annotated[bool, "Whether to use few-shot examples"]
    few_shot_examples: Annotated[
        List[Dict[str, Any]], "List of examples for few-shot learning"
    ]

    # Workflow tracking
    current_node: Annotated[str, "Current node in the graph workflow"]
    current_signal: Annotated[
        Optional[str], "Current signal being processed in individual mode"
    ]
    messages: Annotated[List[Any], "History of LLM interactions"]

    # Results
    label: Annotated[Optional[str], "Classification result (Credible/Fake)"]
    confidence: Annotated[Optional[float], "Confidence score of classification"]
    explanation: Annotated[Optional[str], "Reasoning for classification"]
    credibility_signals: Annotated[Dict[str, Any], "Detected credibility signals"]
    signals_critiques: Annotated[
        Dict[str, Any], "Critic's critiques of detected credibility signals"
    ]
    critic_decision: Annotated[
        Optional[str], "Critic's decision on need for additional analysis"
    ]
    critic_explanation: Annotated[Optional[str], "Explanation of the critic's decision"]
    followup_signals_analysis: Annotated[
        Optional[Dict[str, Any]], "Results from any followup analysis on signals"
    ]
    topic: Annotated[Optional[str], "Topic of the article"]
    classification_prompt: Annotated[Optional[str], "Prompt used for classification"]
    feature_selection: Annotated[
        Optional[Dict[str, Any]], "Features selected for the model"
    ]
    use_rag: Annotated[
        Optional[str], "Whether to use the RAG chain"
    ]
