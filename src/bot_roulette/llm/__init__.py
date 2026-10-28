"""The model: the LLM every call goes through (core.py), OpenRouter's
free checks of an API key and a model id (openrouter.py), and the same
checks for any other OpenAI-compatible server (openai_compatible.py).

The bottom layer: it knows nothing about personas.
"""

from .core import LLM, Cancelled, UnusableOutput, complete, extract_json, set_cancel_event
from .openai_compatible import (
    ServerUnavailable,
    check_server_model,
    is_served_model,
    server_models,
)
from .openrouter import (
    OpenRouterUnavailable,
    check_api_key,
    check_model,
    is_known_model,
    model_catalog,
)

__all__ = [
    "LLM",
    "Cancelled",
    "OpenRouterUnavailable",
    "ServerUnavailable",
    "UnusableOutput",
    "check_api_key",
    "check_model",
    "check_server_model",
    "complete",
    "extract_json",
    "is_known_model",
    "is_served_model",
    "model_catalog",
    "server_models",
    "set_cancel_event",
]
