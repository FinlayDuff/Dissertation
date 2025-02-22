# experiments.py
from typing import TypedDict, Dict, List, Optional


class ModelConfig(TypedDict):
    model_name: str
    temperature: float
    local: bool
    timeout: int


class SignalConfig(TypedDict):
    enabled: bool
    use_bulk: bool
    signals_to_detect: List[str]


class EvaluationConfig(TypedDict):
    few_shot: bool
    few_shot_examples: Optional[List[Dict]]
    metrics: List[str]


class ExperimentConfig(TypedDict):
    description: str
    classification_model: ModelConfig
    signals_model: ModelConfig
    critic_model: ModelConfig
    signals: SignalConfig
    evaluation: EvaluationConfig
    max_concurrency: int


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "baseline_gpt4": {
        "description": "All GPT-4 baseline experiment",
        "classification_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": False,
            "use_bulk": False,
            "signals_to_detect": [
                "bias",
                "emotion",
                "source_credibility",
                "language_complexity",
            ],
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "baseline_claude_3_5_haiku": {
        "description": "All Claude 3.5 Haiku baseline experiment",
        "classification_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": False,
            "use_bulk": False,
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "individual_signals_claude_3_5_haiku": {
        "description": "All Claude 3.5 Haiku experiment with individual signals",
        "max_concurrency": 5,
        "classification_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": False,
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "individual_signals_gpt4": {
        "description": "All GPT-4 experiment with individual signals",
        "classification_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": False,
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "bulk_signals_gpt4": {
        "description": "All GPT-4 experiment with bulk signals",
        "classification_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "bulk_signals_claude_3_5_haiku": {
        "description": "All Claude 3.5 Haiku experiment with bulk signals",
        "max_concurrency": 5,
        "classification_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
    "mixed_models": {
        "description": "Using different models for different tasks",
        "max_concurrency": 2,
        "classification_model": {
            "model_name": "claude-3-sonnet",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "gpt-4",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.1,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
        },
        "evaluation": {
            "few_shot": False,
            "few_shot_examples": None,
            "metrics": ["accuracy", "precision", "recall", "f1"],
        },
    },
}
