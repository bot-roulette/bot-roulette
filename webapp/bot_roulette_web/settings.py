"""The model, where it runs and its API key, as the web app holds them.

The model runs on OpenRouter, or on your own OpenAI-compatible server
(local, like Ollama, LM Studio, vLLM or llama.cpp, or hosted) when
LLM_BASE_URL is set. The key is OPENROUTER_API_KEY, or LLM_API_KEY for
your server, which may need none.

All of them live in the home folder's .env (see home.py), and the ⚙
settings panel changes them while the server runs: the model and the
key are read at CALL time, never frozen at import, and a new server is
applied to the LLM that chats and creation jobs already hold (see
save_settings). The server starts without them, so that the panel can
set them the first time. A key only ever goes to the server it was
saved for: changing the address drops your server's saved key, and
the OpenRouter key is never sent anywhere but OpenRouter.

.env wins over variables already set in the OS environment: a
system-wide variable with a generic name (MODEL is a common one, e.g. a
laptop's model number) must not silently replace what .env — or the
settings panel — says.
"""

import os
import re
import threading

from dotenv import load_dotenv, set_key

from bot_roulette import LLM

from .home import ENV_PATH


def load_env() -> None:
    """Load the home folder's .env into the environment; .env wins."""
    load_dotenv(ENV_PATH, override=True)


class NotConfigured(RuntimeError):
    """The model, or the OpenRouter API key it needs, is not set."""


def server_url() -> str:
    """Your own OpenAI-compatible server (LLM_BASE_URL), or "" when the
    model runs on OpenRouter."""
    return os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")


def server_api_key() -> str:
    """Your server's key (LLM_API_KEY), read at call time: "" if it
    needs none."""
    return os.environ.get("LLM_API_KEY", "").strip()


def _key_name() -> str:
    return "LLM_API_KEY" if server_url() else "OPENROUTER_API_KEY"


def current_model() -> str:
    """The model every call uses, read at call time (see the module notes)."""
    model = os.environ.get("MODEL", "").strip()
    if not model:
        raise NotConfigured(
            "MODEL is not set (see .env, or the web app's settings panel)"
        )
    return model


def current_api_key() -> str:
    """The OpenRouter API key, read at call time (see the module notes)."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise NotConfigured(
            "OPENROUTER_API_KEY is not set (see .env, or the web app's settings panel)"
        )
    return api_key


def is_configured() -> bool:
    """True when the model is set, and so is the OpenRouter API key
    unless the model runs on your own server."""
    try:
        current_model()
        if not server_url():
            current_api_key()
    except NotConfigured:
        return False
    return True


def _llm_on(url: str) -> LLM:
    """The LLM for a model on ``url``: your server's address, or "" for
    OpenRouter. ValueError for a malformed address."""
    # Key and model go in as providers, resolved before EVERY request: a
    # change saved from the settings panel reaches the LLM that chats and
    # creation jobs already hold, and a missing value fails the first
    # call (NotConfigured) instead of the server's start.
    if not url:
        return LLM.openrouter(api_key=current_api_key, model=current_model)
    return LLM.openai_compatible(url, model=current_model, api_key=server_api_key)


def make_llm() -> LLM:
    """The LLM for where .env says the model runs."""
    try:
        return _llm_on(server_url())
    except ValueError as e:
        raise RuntimeError(f"LLM_BASE_URL in .env: {e}") from None


_SETTINGS_LOCK = threading.Lock()

# Values are written into .env unquoted, so they must read back as
# themselves: printable ASCII, no spaces (" #" would start a comment),
# no quotes, no backslash. Real model ids and OpenRouter keys are far
# narrower than this.
_ENV_VALUE_RE = re.compile(r"[!-~]+")
_ENV_UNSAFE = frozenset("\"'#\\")


def _hint(api_key: str) -> str:
    """A key as the settings panel may show it: its last 4 characters."""
    if not api_key:
        return ""
    if len(api_key) >= 16:
        return f"…{api_key[-4:]}"
    return "(set)"  # 4 characters would give away most of a short key


def settings_summary() -> dict:
    """Where the model runs ("server": your server's address, or "" for
    OpenRouter), the model, and hints of the keys, never the keys
    themselves: the one in use, and the OpenRouter one, kept for when
    the model runs there."""
    url = server_url()
    model = os.environ.get("MODEL", "").strip()
    api_key = os.environ.get(_key_name(), "").strip()
    return {
        "server": url,
        "model": model,
        "api_key_hint": _hint(api_key),
        "openrouter_key_hint": _hint(os.environ.get("OPENROUTER_API_KEY", "").strip()),
        "configured": bool(model and (api_key or url)),
    }


def save_settings(
    *,
    server: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    llm: LLM | None = None,
) -> None:
    """Write the settings into .env and apply them to this process at
    once; None leaves a setting as it is. ``server`` is your server's
    address, or "" for OpenRouter; ``api_key`` is the key for wherever
    the model runs after the change. Check new values first (see the
    server's POST /settings): this only stores them.

    A key only goes to the server it was saved for: a new address drops
    your server's saved key, unless ``api_key`` gives the new one.

    ``llm``, the one the app's chats and creation jobs hold, moves to a
    new server in the same step: its client is built before anything is
    written (a failure saves nothing) and swapped in right after the
    environment changes, so a call can't pair the new model with the old
    server. A call already under way ends where it started.
    """
    target = server_url() if server is None else server.strip().rstrip("/")
    moving = target != server_url()
    updates = {}
    if moving:
        updates["LLM_BASE_URL"] = target
        updates["LLM_API_KEY"] = ""
    if model is not None:
        updates["MODEL"] = model
    if api_key is not None:
        updates["LLM_API_KEY" if target else "OPENROUTER_API_KEY"] = api_key
    for name, value in updates.items():
        # Empty is allowed: it clears a value, and still wins over the
        # same variable in the OS environment (leaving it out wouldn't).
        if value and (not _ENV_VALUE_RE.fullmatch(value) or _ENV_UNSAFE & set(value)):
            raise ValueError(f"{name} has characters that can't be stored in .env")
    fresh = _llm_on(target) if llm is not None and moving else None
    with _SETTINGS_LOCK:
        for name, value in updates.items():
            set_key(ENV_PATH, name, value, quote_mode="never")
        os.environ.update(updates)
        if fresh is not None:
            llm.client, llm.json_format, llm.reasoning_off = (
                fresh.client, fresh.json_format, fresh.reasoning_off
            )
