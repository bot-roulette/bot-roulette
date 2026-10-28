"""The web app with its model on your own OpenAI-compatible server
(LLM_BASE_URL in .env; a local one, faked): no OpenRouter key needed, the
settings panel checks against the models the server lists, and it can
move the model to another server, or to OpenRouter."""

import pytest


def test_a_model_is_all_it_needs(local_web):
    view = local_web.client.get("/settings").json()
    assert view == {"server": local_web.local_url, "model": "", "api_key_hint": "",
                    "openrouter_key_hint": "", "configured": False, "model_known": None}
    r = local_web.client.post("/settings", json={"model": "llama3"})
    assert r.status_code == 400 and "no model called 'llama3'" in r.json()["detail"]

    # Without a tag, a name is Ollama's ":latest".
    r = local_web.client.post("/settings", json={"model": "gemma3"})
    assert r.status_code == 200
    assert r.json() == {"server": local_web.local_url, "model": "gemma3", "api_key_hint": "",
                        "openrouter_key_hint": "", "configured": True, "model_known": True}
    env = (local_web.home / ".env").read_text(encoding="utf-8")
    assert "MODEL=gemma3\n" in env and "KEY" not in env
    assert local_web.client.get("/personas").json()["configured"] is True

    llm = local_web.server.llm
    llm.client._refresh_api_key()
    assert str(llm.client.base_url) == local_web.local_url + "/"
    assert llm.resolve_model() == "gemma3"
    # A placeholder key: never the OpenAI key set system-wide.
    assert llm.client.auth_headers == {"Authorization": "Bearer no-key"}
    assert llm.json_format["type"] == "json_schema" and llm.reasoning_off == {}


def test_suggestions_are_the_models_the_server_lists(local_web):
    r = local_web.client.get("/settings/models")
    assert [m["id"] for m in r.json()["models"]] == local_web.local_models
    local_web.openrouter.local_down = True
    try:
        r = local_web.client.get("/settings/models")
        assert r.status_code == 502 and "couldn't reach" in r.json()["detail"]
        r = local_web.client.post("/settings", json={"model": "qwen2.5vl:7b"})
        assert r.status_code == 502 and "nothing was saved" in r.json()["detail"]
        assert local_web.client.get("/settings").json()["model_known"] is None
    finally:
        local_web.openrouter.local_down = False
    assert local_web.openrouter.calls["models"] == local_web.openrouter.calls["key"] == 0


def test_a_key_is_checked_with_the_server_and_sent_to_it(local_web):
    local_web.openrouter.local_key = "lm-secret-key-1234"
    try:
        r = local_web.client.post("/settings", json={"api_key": "wrong-key"})
        assert r.status_code == 400 and "rejected" in r.json()["detail"]
        r = local_web.client.post("/settings", json={"api_key": "lm-secret-key-1234"})
        assert r.status_code == 200 and r.json()["api_key_hint"] == "\u20261234"
        env = (local_web.home / ".env").read_text(encoding="utf-8")
        assert "LLM_API_KEY=lm-secret-key-1234\n" in env and "OPENROUTER_API_KEY" not in env
        llm = local_web.server.llm
        llm.client._refresh_api_key()
        assert llm.client.auth_headers == {"Authorization": "Bearer lm-secret-key-1234"}
    finally:
        local_web.openrouter.local_key = ""


def test_the_panel_moves_the_model_between_servers(local_web):
    """Where the model runs is a setting like the others; a key only goes
    to the server it was saved for."""
    client, fake, llm = local_web.client, local_web.openrouter, local_web.server.llm
    local, other = local_web.local_url, local_web.other_url
    # The starting point: a model and a key saved for the local server.
    r = client.post("/settings", json={"model": "gemma3", "api_key": "lm-secret-key-1234"})
    assert r.status_code == 200 and client.get("/settings").json()["server"] == local

    # Suggestions for an address being typed: asked without the saved key.
    r = client.get("/settings/models", params={"server": other})
    assert [m["id"] for m in r.json()["models"]] == local_web.other_models
    assert fake.server_auth[-1] == (other, None)
    client.get("/settings/models", params={"server": local + "/"})
    assert fake.server_auth[-1] == (local, "Bearer lm-secret-key-1234")
    r = client.get("/settings/models", params={"server": "localhost:11434"})
    assert r.status_code == 400 and "not an http" in r.json()["detail"]

    # Moving needs a model the other server has: the current one isn't.
    r = client.post("/settings", json={"server": other})
    assert r.status_code == 400 and "no model called 'gemma3'" in r.json()["detail"]
    assert client.get("/settings").json()["server"] == local
    # A switch that fails saves nothing, and the chats' LLM stays put.
    env_before = (local_web.home / ".env").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="not an http"):
        local_web.server.save_settings(server="http://[::1/v1", model="x", llm=llm)
    assert (local_web.home / ".env").read_text(encoding="utf-8") == env_before
    assert str(llm.client.base_url) == local + "/"

    r = client.post("/settings", json={"server": other + "/", "model": "llama-3.2-vision"})
    assert r.status_code == 200
    assert r.json() == {"server": other, "model": "llama-3.2-vision", "api_key_hint": "",
                        "openrouter_key_hint": "", "configured": True, "model_known": True}
    env = (local_web.home / ".env").read_text(encoding="utf-8")
    assert f"LLM_BASE_URL={other}\n" in env and "LLM_API_KEY=\n" in env
    # The new server never saw the old key, and the chats' LLM moved with the setting.
    assert {auth for server, auth in fake.server_auth if server == other} == {None}
    llm.client._refresh_api_key()
    assert str(llm.client.base_url) == other + "/"
    assert llm.client.auth_headers == {"Authorization": "Bearer no-key"}

    # OpenRouter needs its key, and a model from its catalog.
    r = client.post("/settings", json={"server": "", "model": "z-ai/glm-5.2"})
    assert r.status_code == 400 and "needs your API key" in r.json()["detail"]
    r = client.post("/settings", json={"server": "", "model": "z-ai/glm-5.2", "api_key": local_web.good_key})
    assert r.status_code == 200
    assert r.json() == {"server": "", "model": "z-ai/glm-5.2", "api_key_hint": "\u2026abcd",
                        "openrouter_key_hint": "\u2026abcd", "configured": True, "model_known": True}
    assert "LLM_BASE_URL=\n" in (local_web.home / ".env").read_text(encoding="utf-8")
    llm.client._refresh_api_key()
    assert str(llm.client.base_url) == "https://openrouter.ai/api/v1/"
    assert llm.json_format == {"type": "json_object"}
    assert llm.reasoning_off == {"reasoning": {"effort": "none"}}
    r = client.get("/settings/models", params={"server": ""})
    assert [m["id"] for m in r.json()["models"]] == [m["id"] for m in local_web.catalog]

    # Back on your server: the OpenRouter key stays for next time, and goes nowhere else.
    r = client.post("/settings", json={"server": local, "model": "qwen2.5vl:7b"})
    assert r.status_code == 200
    assert (r.json()["api_key_hint"], r.json()["openrouter_key_hint"]) == ("", "\u2026abcd")
    llm.client._refresh_api_key()
    assert str(llm.client.base_url) == local + "/"
    assert llm.json_format["type"] == "json_schema" and llm.reasoning_off == {}
    sent = [str(auth) for _, auth in fake.server_auth]
    assert not any(local_web.good_key in auth or "sk-from-the-os" in auth for auth in sent)
