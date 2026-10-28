"""Checks for any OpenAI-compatible server, local (Ollama, LM Studio,
vLLM, llama.cpp...) or hosted: which models it serves, and whether it
takes a key. One free call (no tokens are spent): GET /models.
"""

import re

import httpx

from .openrouter import _KEY_RE

# An http(s) address without a query or a fragment: the OpenAI client
# appends the API's paths (/models, /chat/completions) to it.
_URL_RE = re.compile(r"https?://[^\s/?#]+(?:/[^\s?#]*)?")


def _check_url(base_url: str) -> None:
    # The regex keeps out what makes no sense as an API's address; httpx
    # then rejects what it can't parse (a bad port, an unclosed "[").
    try:
        valid = bool(_URL_RE.fullmatch(base_url)) and bool(httpx.URL(base_url).host)
    except httpx.InvalidURL:
        valid = False
    if not valid:
        raise ValueError(f"not an http:// or https:// address: {base_url!r}")


class ServerUnavailable(RuntimeError):
    """The server gave no usable answer to a settings check (not running,
    unreachable, an unexpected reply). Says nothing about whether the
    value being checked is valid."""


def server_models(base_url: str, api_key: str = "") -> list[str]:
    """The ids of the models the server at ``base_url`` serves.
    ValueError for a malformed address, or if the server rejects
    ``api_key`` (sent only when given)."""
    _check_url(base_url)
    if api_key and not _KEY_RE.fullmatch(api_key):
        raise ValueError(
            "that doesn't look like an API key (spaces or stray characters)"
        )
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        response = httpx.get(url, headers=headers, timeout=10)
    except httpx.HTTPError as e:
        raise ServerUnavailable(f"couldn't reach {url} ({e})") from e
    if response.status_code in (401, 403):
        raise ValueError(
            "the server rejected this API key" if api_key else "the server wants an API key"
        )
    if response.status_code != 200:
        raise ServerUnavailable(f"{url} answered HTTP {response.status_code}")
    try:
        entries = response.json()["data"]
    except (ValueError, KeyError, TypeError) as e:
        raise ServerUnavailable(f"{url} sent an unreadable answer") from e
    return [
        m["id"] for m in (entries if isinstance(entries, list) else [])
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    ]


def is_served_model(base_url: str, model: str, api_key: str = "") -> bool:
    """Whether the server serves ``model``. A name without a tag also
    matches the same name tagged ":latest", Ollama's default tag."""
    models = server_models(base_url, api_key)
    return model in models or f"{model}:latest" in models


def check_server_model(base_url: str, model: str, api_key: str = "") -> None:
    """ValueError unless the server serves ``model``."""
    if not is_served_model(base_url, model, api_key):
        raise ValueError(f"the server has no model called {model!r}")
