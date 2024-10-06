from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def get_llm_from_model_name(model_name: str):
    # Check if the model name suggests it's an Anthropic model
    if "claude" in model_name:
        return ChatAnthropic(model=model_name, temperature=0)

    # Check if it's an OpenAI model
    elif "gpt" in model_name:
        return ChatOpenAI(model=model_name, temperature=0)

    # Add more model providers if needed
    else:
        raise ValueError(f"Unknown model provider for model: {model_name}")
