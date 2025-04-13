from transformers import pipeline
import os
from typing import Callable, Dict
from core.state import State

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def bias_classifier(state: State):
    """
    Classifies bias in article content using distilroberta-bias model.
    Handles long texts through automatic truncation and striding.
    """
    pipe = pipeline(
        "text-classification",
        model="valurank/distilroberta-bias",
        truncation=True,
        padding=True,
        max_length=512,
        top_k=None,  # Returns all scores instead of using deprecated return_all_scores
    )

    article_content = state.get("article_content", "")
    results = pipe(
        article_content,
        batch_size=8,
        stride=128,  # Removed invalid 'overlap' parameter
    )

    # Average scores across chunks if multiple chunks were processed
    if isinstance(results[0], list):
        scores = [chunk[0]["score"] for chunk in results]
        avg_score = sum(scores) / len(scores)
        label = "BIASED" if avg_score > 0.5 else "UNBIASED"

        return {
            "analysis_type": "roBERTa-bias-classification",
            "results": {
                "label": label,
                "score": float(avg_score),
                "num_chunks": len(results),
            },
        }

    # Single result case
    return {
        "analysis_type": "roBERTa-bias-classification",
        "results": {
            "label": "BIASED" if results[0]["score"] > 0.5 else "UNBIASED",
            "score": float(results[0]["score"]),
            "num_chunks": 1,
        },
    }


# Map signal names to their corresponding analysis functions
FOLLOWUP_TOOLS: Dict[str, Callable[[State], Dict]] = {
    "bias": {
        "method": bias_classifier,
        "description": "A roBERTa model for bias classification",
    },
    "evidence": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "inference": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "polarising_language": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "document_citation": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "informal_tone": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "explicitly_unverified_claims": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "personal_perspective": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "emotional_valence": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "call_to_action": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "expert_citation": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "clickbait": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "incorrect_spelling": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "misleading_about_content": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "incivility": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "impoliteness": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "sensationalism": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "source_credibility": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
    "reported_by_other_sources": {
        "method": "llm",
        "description": "An LLM prompted to reclasify the credibility signal",
    },
}
