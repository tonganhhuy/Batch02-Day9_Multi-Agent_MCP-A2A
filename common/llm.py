"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def validate_openrouter_config() -> tuple[str, str]:
    """Validate configuration early so auth errors do not surface deep in A2A."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()

    if not api_key or api_key == "your_key_here":
        raise RuntimeError("OPENROUTER_API_KEY is missing in .env")
    if not api_key.startswith("sk-or-"):
        raise RuntimeError(
            "OPENROUTER_API_KEY has an invalid format. "
            "Create a valid OpenRouter key beginning with 'sk-or-' and update .env."
        )
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is missing in .env")
    return api_key, model


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    api_key, model = validate_openrouter_config()
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=800,
        temperature=0.3,
    )
