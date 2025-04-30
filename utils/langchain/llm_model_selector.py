"""
LLM Model Selector Module

This module provides functionality for initializing and managing different LLM providers
(OpenAI, Anthropic, Cohere, HuggingFace) with retry logic and caching.

The module handles:
- Provider-specific model initialization
- Rate limit retries
- Model caching
- Local vs remote model deployment
"""

import time
import random
from functools import wraps, lru_cache
from typing import Any, Callable, Dict, TypeVar, cast
import torch
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config.experiments import ModelConfig

# Use MPS if available on Mac, otherwise CPU
device = 0 if torch.backends.mps.is_available() else -1

F = TypeVar("F", bound=Callable[..., Any])

from openai import APIConnectionError, RateLimitError
import httpx
import logging
from duckduckgo_search.duckduckgo_search import RatelimitException

logger = logging.getLogger(__name__)


def retry_on_api_exceptions(
    max_retries: int = 10, backoff_factor: float = 1.0
) -> Callable[[F], F]:
    """
    Decorator that retries on API rate limits and transient connection errors
    using exponential backoff.

    Retries on:
    - openai.error.RateLimitError
    - openai.APIConnectionError
    - httpx.ConnectError
    - TLS/SSL EOF errors (by string matching)

    Args:
        max_retries: Maximum number of retries
        backoff_factor: Base delay factor

    Returns:
        Wrapped function with retry logic
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    logging.error(f"Error occurred: {e}")
                    error_str = str(e).lower()
                    retriable = any(
                        substr in error_str
                        for substr in [
                            "rate_limit_error",
                            "rate limit exceeded",
                            "tls",
                            "ssl",
                            "connection reset",
                            "eof",
                            "connection error",
                            "apiconnectionerror",
                            "connecterror",
                        ]
                    ) or isinstance(
                        e,
                        (
                            RateLimitError,
                            APIConnectionError,
                            httpx.ConnectError,
                            RatelimitException,
                        ),
                    )

                    if retriable:
                        retries += 1
                        sleep_time = backoff_factor * (2**retries) + random.uniform(
                            0, 1
                        )
                        logging.info(
                            f"[Retry {retries}/{max_retries}] Retriable error: {error_str}. "
                            f"Retrying in {sleep_time:.2f} seconds..."
                        )
                        time.sleep(sleep_time)
                    else:
                        raise e

            raise Exception(f"Max retries ({max_retries}) exceeded")

        return cast(F, wrapper)

    return decorator


class CustomChatOpenAI(ChatOpenAI):
    @retry_on_api_exceptions(max_retries=10, backoff_factor=1.0)
    def _generate(self, *args, **kwargs):
        return super()._generate(*args, **kwargs)


# Provider initialization functions
PROVIDER_MAPPING: Dict[str, Callable[[ModelConfig], Any]] = {
    "claude": lambda c: ChatAnthropic(
        model=c.model_name,
        temperature=c.temperature,
    ),
    "gpt": lambda c: CustomChatOpenAI(
        model=c.model_name,
        temperature=c.temperature,
    ),
    "cohere": lambda c: ChatCohere(
        model=c.model_name,
        temperature=c.temperature,
    ),
    "hf": lambda c: ChatHuggingFace(
        model=c.model_name,
        temperature=c.temperature,
    ),
}


@lru_cache(maxsize=None)
@retry_on_api_exceptions()
def get_llm_from_model_name(config: ModelConfig) -> Any:
    """
    Initialize and return an LLM instance based on the provided configuration.

    This function handles different model providers (OpenAI, Anthropic, etc.) and
    supports both local and remote model deployment. Results are cached to avoid
    redundant initialization.

    Args:
        config: Configuration dictionary containing:
            - model_name: Name of the model to initialize
            - temperature: Sampling temperature for generation
            - local: Whether to run locally (for supported models)
            - Other provider-specific settings

    Returns:
        LLM instance for the specified model provider, ready for use in the langgraph system:
            ChatAnthropic: For Claude models
            ChatOpenAI: For GPT models
            ChatCohere: For Cohere models
            ChatHuggingFace: For HuggingFace and LLaMA models

    Raises:
        ValueError: If the model provider is unknown or configuration is invalid

    Example:
        config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0,
            "local": False
        }
        llm = get_llm_from_model_name(config)
    """
    # Try standard providers based on model name prefix
    for prefix, provider_func in PROVIDER_MAPPING.items():
        if config.model_name.startswith(prefix):
            return provider_func(config)

    # Special handling for LLaMA models
    if config.model_name.startswith("llama"):
        if config.local:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
            local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            llm = HuggingFacePipeline(pipeline=local_pipeline)
        else:
            llm = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-2-7b-chat-hf",
                temperature=config.temperature,
            )
        return ChatHuggingFace(llm=llm)

    raise ValueError(f"Unknown model provider for model: {config.model_name}")
