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
    "zero_shot_gpt4": {
        "description": "GPT-4 zero_shot experiment",
        "max_concurrency": 5,
        "classification_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
    },
    "zero_shot_gpt35_turbo": {
        "description": "GPT-3.5 turbo zero_shot experiment",
        "max_concurrency": 5,
        "classification_model": {
            "model_name": "gpt-3.5-turbo-0125",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
    },
    "few_shot_gpt4": {
        "description": "All GPT-4 few shot experiment",
        "max_concurrency": 5,
        "classification_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "few_shot": True,
    },
    "zero_shot_claude": {
        "description": "Claude 3.5 Haiku zero_shot experiment",
        "classification_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
    },
    "bulk_signals_gpt4": {
        "description": "All GPT-4 experiment with bulk signals",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
        },
    },
    "bulk_signals_condensed_gpt4": {
        "description": "All GPT-4 experiment with bulk signals condensed",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "condensed": True,
        },
    },
    "bulk_signals_condensed_critic_gpt4": {
        "description": "All GPT-4 experiment with bulk signals condensed and critic",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "condensed": True,
        },
    },
    "bulk_signals_condensed_critic_followup_gpt4": {
        "description": "All GPT-4 experiment with bulk signals condensed, critic and followup",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "rag_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "condensed": True,
            "followup": True,
        },
    },
    "bulk_signals_condensed_critic_followup_selector_gpt4": {
        "description": "All GPT-4 experiment with bulk signals condensed, critic and followup",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "rag_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "condensed": True,
            "followup": True,
            "feature_selector": True,
        },
    },
    "bulk_signals_critic_followup_selector_gpt4": {
        "description": "All GPT-4 experiment with bulk signals condensed, critic and followup",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "rag_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "condensed": False,
            "followup": True,
            "feature_selector": True,
        },
    },
    "bulk_signals_condensed_critic_followup_selector_no_rag_gpt4": {
        "description": "All GPT-4 experiment with bulk signals condensed, critic and followup",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "rag_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "condensed": True,
            "followup": True,
            "feature_selector": True,
            "use_rag": False,
        },
    },
    "bulk_signals_critic_followup_selector_gpt4": {
        "description": "All GPT-4 experiment with bulk signals, critic and followup",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "rag_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "followup": True,
            "feature_selector": True,
        },
    },
    "bulk_signals_followup_gpt4": {
        "description": "All GPT-4 experiment with bulk signals and followup",
        "max_concurrency": 5,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "rag_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
            "critic": True,
            "followup": True,
        },
    },
    "bulk_signals_few_shot_gpt4": {
        "description": "All GPT-4 experiment with bulk signals and few-shot",
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals": {
            "enabled": True,
            "use_bulk": True,
        },
        "few_shot": True,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
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
    "bulk_signals_claude_3_5_sonnet": {
        "description": "All Claude 3.5 Sonnet experiment with bulk signals",
        "max_concurrency": 5,
        "classification_model": {
            "model_name": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "signals_model": {
            "model_name": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "followup_model": {
            "model_name": "claude-3-5-sonnet-20241022",
            "temperature": 0.0,
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
    "bulk_signals_gpt4_claude3_5": {
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
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "claude-3-5-haiku-20241022",
            "temperature": 0.0,
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
            "temperature": 0.0,
            "local": False,
            "timeout": 300,
        },
        "critic_model": {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
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
