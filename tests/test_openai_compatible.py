import httpx
import pytest

from bot_roulette.llm import openai_compatible
from bot_roulette.llm.openai_compatible import (
    ServerUnavailable,
    check_server_model,
    is_served_model,
    server_models,
)

URL = "http://localhost:11434/v1"
MODELS = {"object": "list", "data": [{"id": "qwen2.5vl:7b"}, {"id": "gemma3:latest"}, {"object": "no id"}]}


@pytest.fixture
def server(monkeypatch):
    """A server's GET /models, faked; records the calls."""
    calls = []
    state = {"down": False, "key": "", "status": 200, "body": MODELS}

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        if state["down"]:
            raise httpx.ConnectError("connection refused")
        if state["key"] and (headers or {}).get("Authorization") != f"Bearer {state['key']}":
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(state["status"], json=state["body"])

    monkeypatch.setattr(openai_compatible.httpx, "get", fake_get)
    return calls, state


def test_the_models_a_server_lists(server):
    calls, _ = server
    assert server_models(URL + "/") == ["qwen2.5vl:7b", "gemma3:latest"]
    assert calls == [(URL + "/models", None)]


def test_a_name_without_a_tag_is_its_latest(server):
    assert is_served_model(URL, "qwen2.5vl:7b") and is_served_model(URL, "gemma3")
    with pytest.raises(ValueError, match="no model called 'qwen2.5vl'"):
        check_server_model(URL, "qwen2.5vl")


def test_a_key_is_sent_only_when_given(server):
    calls, state = server
    state["key"] = "lm-key"
    with pytest.raises(ValueError, match="wants an API key"):
        server_models(URL)
    with pytest.raises(ValueError, match="rejected"):
        server_models(URL, "wrong-key")
    assert server_models(URL, "lm-key")
    assert calls[-1] == (URL + "/models", {"Authorization": "Bearer lm-key"})
    with pytest.raises(ValueError, match="doesn't look like"):
        server_models(URL, "has space")
    assert len(calls) == 3  # the malformed key never left


@pytest.mark.parametrize("trouble", [{"down": True}, {"status": 500}, {"body": {"no": "data"}}])
def test_a_server_that_cant_answer_is_not_a_verdict(server, trouble):
    _, state = server
    state.update(trouble)
    with pytest.raises(ServerUnavailable):
        server_models(URL)


@pytest.mark.parametrize("url", [
    "localhost:11434/v1", "ftp://host/v1", "http://host/v1?x=1", "http://host/v1#top",
    "http://[::1/v1", "http://:80/v1",
])
def test_a_malformed_address_never_reaches_the_network(server, url):
    calls, _ = server
    with pytest.raises(ValueError, match="not an http"):
        server_models(url)
    assert calls == []
