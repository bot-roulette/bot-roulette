import httpx
import pytest

from bot_roulette.llm import openrouter
from bot_roulette.llm.openrouter import (
    OpenRouterUnavailable,
    catalog_id,
    check_api_key,
    check_model,
    model_catalog,
)

CATALOG = [{"id": "x/y", "name": "Y"}, {"id": "x/y:free", "name": "Y (free)"}, {"id": "openrouter/auto"}]


@pytest.fixture
def openrouter_api(monkeypatch):
    """Fakes OpenRouter's GET /key and GET /models; records the calls."""
    calls = []
    state = {"down": False, "good_key": "sk-or-v1-good"}

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        if state["down"]:
            raise httpx.ConnectError("network is down")
        if url.endswith("/models"):
            return httpx.Response(200, json={"data": CATALOG})
        ok = (headers or {}).get("Authorization") == f"Bearer {state['good_key']}"
        return httpx.Response(200 if ok else 401, json={"data": {"label": "k"}} if ok else {})

    monkeypatch.setattr(openrouter.httpx, "get", fake_get)
    monkeypatch.setattr(openrouter, "_catalog", None)
    return calls, state


@pytest.mark.parametrize(
    "model, entry",
    [("a/b", "a/b"), ("a/b:nitro", "a/b"), ("a/b:free:nitro", "a/b:free"),
     ("a/b:nitro:exacto", "a/b"), ("a/b:typo", "a/b:typo")],
)
def test_catalog_id_follows_openrouters_rule(model, entry):
    assert catalog_id(model) == entry


def test_check_api_key(openrouter_api):
    calls, _ = openrouter_api
    assert check_api_key("sk-or-v1-good") == {"label": "k"}
    assert calls[0] == ("https://openrouter.ai/api/v1/key", {"Authorization": "Bearer sk-or-v1-good"})
    with pytest.raises(ValueError, match="rejected"):
        check_api_key("sk-or-v1-wrong")


@pytest.mark.parametrize("key", ["has space", "new\nline", "\u00e8\u00e8"])
def test_malformed_keys_never_reach_the_network(openrouter_api, key):
    calls, _ = openrouter_api
    with pytest.raises(ValueError, match="doesn't look like"):
        check_api_key(key)
    assert calls == []


def test_check_model(openrouter_api):
    for model in ("x/y", "x/y:free", "x/y:nitro", "openrouter/auto"):
        check_model(model)
    for model in ("7440", "x/y:typo"):
        with pytest.raises(ValueError, match="no model called"):
            check_model(model)


def test_the_catalog_is_cached_but_checks_ask_afresh(openrouter_api):
    calls, _ = openrouter_api
    model_catalog()
    model_catalog()
    assert len(calls) == 1
    check_model("x/y")
    assert len(calls) == 2


def test_unreachable_openrouter_is_not_a_verdict(openrouter_api):
    _, state = openrouter_api
    state["down"] = True
    with pytest.raises(OpenRouterUnavailable):
        check_api_key("sk-or-v1-good")
    with pytest.raises(OpenRouterUnavailable):
        check_model("x/y")
