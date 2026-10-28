"""OpenRouter helpers: check an API key, list and check model ids.

All of these are free calls (no tokens are spent): GET /key answers for
a key, GET /models is OpenRouter's public catalog. Use them to validate
settings before a program starts relying on them.
"""

import re
import time

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# A key travels in an HTTP header: printable ASCII, no spaces.
_KEY_RE = re.compile(r"[!-~]+")


class OpenRouterUnavailable(RuntimeError):
    """OpenRouter gave no usable answer to a settings check (network
    error, outage, rate limit, unreadable body). Says nothing about
    whether the value being checked is valid."""


# Routing variants are accepted on any model id but never listed in the
# catalog; catalog variants (:free, :batch...) are entries of their own.
# https://openrouter.ai/docs/guides/routing/model-variants
_ROUTING_VARIANTS = frozenset({"nitro", "floor", "exacto", "online"})

_CATALOG_TTL = 600  # seconds the catalog is reused for model suggestions
_catalog: tuple[float, list[dict]] | None = None


def _openrouter_get(path: str, *, api_key: str | None = None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        return httpx.get(f"{OPENROUTER_URL}{path}", headers=headers, timeout=15)
    except httpx.HTTPError as e:
        raise OpenRouterUnavailable(f"couldn't reach OpenRouter ({e})") from e


def _openrouter_data(response: httpx.Response):
    """The "data" field of a successful OpenRouter answer."""
    if response.status_code != 200:
        raise OpenRouterUnavailable(f"OpenRouter answered HTTP {response.status_code}")
    try:
        return response.json()["data"]
    except (ValueError, KeyError, TypeError) as e:
        raise OpenRouterUnavailable("OpenRouter sent an unreadable answer") from e


def check_api_key(api_key: str) -> dict:
    """Ask OpenRouter about ``api_key`` (free: no tokens are spent).
    Returns the key's info (label, limit, usage...); ValueError if
    OpenRouter rejects the key."""
    if not _KEY_RE.fullmatch(api_key):
        raise ValueError(
            "that doesn't look like an API key (spaces or stray characters)"
        )
    response = _openrouter_get("/key", api_key=api_key)
    if response.status_code in (401, 403):
        raise ValueError("OpenRouter rejected this API key")
    data = _openrouter_data(response)
    return data if isinstance(data, dict) else {}


def model_catalog(*, refresh: bool = False) -> list[dict]:
    """OpenRouter's public model catalog as [{"id", "name"}], reused for
    _CATALOG_TTL seconds unless ``refresh``."""
    global _catalog
    if not refresh and _catalog and time.monotonic() - _catalog[0] < _CATALOG_TTL:
        return _catalog[1]
    entries = _openrouter_data(_openrouter_get("/models"))
    models = [
        {"id": m["id"], "name": str(m.get("name") or m["id"])}
        for m in (entries if isinstance(entries, list) else [])
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    ]
    _catalog = (time.monotonic(), models)
    return models


def catalog_id(model: str) -> str:
    """The catalog entry a model id resolves to, by OpenRouter's rule:
    routing variants dropped, the rest kept ("a/b:free:nitro" -> "a/b:free")."""
    base, *suffixes = model.split(":")
    return ":".join([base, *(s for s in suffixes if s not in _ROUTING_VARIANTS)])


def is_known_model(model: str, *, refresh: bool = False) -> bool:
    """Whether OpenRouter's catalog has ``model`` (variants resolved)."""
    return catalog_id(model) in {m["id"] for m in model_catalog(refresh=refresh)}


def check_model(model: str) -> None:
    """ValueError unless OpenRouter's catalog has ``model``. Always asks
    afresh: a cached catalog would reject a model released minutes ago."""
    if not is_known_model(model, refresh=True):
        raise ValueError(f"OpenRouter has no model called {model!r}")
