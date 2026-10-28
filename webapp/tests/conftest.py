import importlib
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

REPO = Path(__file__).resolve().parents[2]

GOOD_KEY = "sk-or-v1-" + "0" * 60 + "abcd"
GOOD_KEY_2 = "sk-or-v1-" + "1" * 60 + "wxyz"
CATALOG = [
    {"id": "openrouter/auto", "name": "Auto Router"},
    {"id": "z-ai/glm-5.2", "name": "GLM 5.2"},
]
# Your own OpenAI-compatible servers, as Ollama would answer: one on this
# machine, and one elsewhere on the network.
LOCAL_URL = "http://localhost:11434/v1"
LOCAL_MODELS = ["qwen2.5vl:7b", "gemma3:latest"]
OTHER_URL = "http://192.168.1.50:8000/v1"
OTHER_MODELS = ["llama-3.2-vision"]


class FakeClient:
    """Stands in for openai.OpenAI behind the server's LLM: scripted
    replies, in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.replies.pop(0), reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class FakeOpenRouter:
    """OpenRouter's GET /key and GET /models, and your own servers' GET
    /models (at LOCAL_URL and OTHER_URL), faked at the httpx level."""

    def __init__(self):
        self.calls = {"models": 0, "key": 0, "local": 0}
        self.models_down = False
        self.local_down = False
        self.local_key = ""  # the key the local server wants, if any
        self.server_auth = []  # (server, Authorization header) of each call to your servers

    def get(self, url, headers=None, timeout=None):
        if url.startswith(OTHER_URL):
            self.server_auth.append((OTHER_URL, (headers or {}).get("Authorization")))
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": m, "object": "model"} for m in OTHER_MODELS
            ]})
        if url.startswith(LOCAL_URL):
            self.calls["local"] += 1
            self.server_auth.append((LOCAL_URL, (headers or {}).get("Authorization")))
            if self.local_down:
                raise httpx.ConnectError("connection refused")
            if self.local_key and (headers or {}).get("Authorization") != f"Bearer {self.local_key}":
                return httpx.Response(401, json={"error": "invalid api key"})
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": m, "object": "model"} for m in LOCAL_MODELS
            ]})
        if url.endswith("/models"):
            self.calls["models"] += 1
            if self.models_down:
                raise httpx.ConnectError("network is down")
            return httpx.Response(200, json={"data": CATALOG})
        self.calls["key"] += 1
        auth = (headers or {}).get("Authorization", "")
        if "down" in auth:
            raise httpx.ConnectError("network is down")
        if auth in (f"Bearer {GOOD_KEY}", f"Bearer {GOOD_KEY_2}"):
            return httpx.Response(200, json={"data": {"label": "test"}})
        return httpx.Response(401, json={"error": {"message": "User not found."}})


# Chat-ready personas the tests talk to. Written for the tests, not
# copied from the shipped personas/ folder, whose personas come and go.
TEST_PERSONAS = {
    "francesca_20260907_064108": "Francesca",
    "monica_20260907_060117": "Monica",
    "simona_20260906_225636": "Simona",
}


def _write_persona(folder, name):
    """A minimal enriched, voice-calibrated persona folder."""
    folder.mkdir(parents=True)
    stem = name.lower()
    (folder / f"{stem}.md").write_text(
        f"=== WHO YOU ARE ===\n{name}, 30, lives in Turin.\n"
        "Languages: Italian (native), English (B1).",
        encoding="utf-8",
    )
    (folder / f"{stem}.yaml").write_text(
        f"name: {name}\nfull_name: {name} Rossi\ncore: {stem}.md\n"
        f"parent: {stem}\nvoice_calibrated: true\n",
        encoding="utf-8",
    )


def _serve(tmp_path_factory, **env):
    """The web app, imported against a fresh home folder (a copy of the
    shipped config, and test personas of its own) with no key and the
    given environment, OpenRouter and a local server faked. The
    environment is restored afterwards."""
    home = tmp_path_factory.mktemp("home")
    shutil.copy(REPO / "config.yaml", home / "config.yaml")
    for folder, name in TEST_PERSONAS.items():
        _write_persona(home / "personas" / folder, name)

    saved_env = dict(os.environ)
    # The tests that want an access code set it themselves.
    for name in ("OPENROUTER_API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "MODEL", "ACCESS_CODE"):
        os.environ.pop(name, None)
    os.environ.update(
        BOT_ROULETTE_HOME=str(home),
        GENERATION_ALLOWED="true", CHAT_MEMORY_ALLOWED="true", **env,
    )

    import bot_roulette.llm.openrouter as openrouter

    fake = FakeOpenRouter()
    real_get, openrouter.httpx.get = openrouter.httpx.get, fake.get
    openrouter._catalog = None
    for name in [m for m in sys.modules if m.startswith("bot_roulette_web")]:
        del sys.modules[name]
    server = importlib.import_module("bot_roulette_web.server")
    from fastapi.testclient import TestClient

    try:
        yield SimpleNamespace(
            server=server, client=TestClient(server.app), home=home,
            openrouter=fake, openrouter_module=openrouter, FakeClient=FakeClient,
            good_key=GOOD_KEY, good_key_2=GOOD_KEY_2, catalog=CATALOG,
            local_url=LOCAL_URL, local_models=LOCAL_MODELS,
            other_url=OTHER_URL, other_models=OTHER_MODELS,
        )
    finally:
        openrouter.httpx.get = real_get
        openrouter._catalog = None
        os.environ.clear()
        os.environ.update(saved_env)


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    """The web app on OpenRouter, with no key and a stray MODEL."""
    yield from _serve(tmp_path_factory, MODEL="7440")


@pytest.fixture(scope="module")
def local_web(tmp_path_factory):
    """The web app on your own server (LLM_BASE_URL), with nothing else
    set but a system-wide OpenAI key, which must never be sent."""
    yield from _serve(tmp_path_factory, LLM_BASE_URL=LOCAL_URL + "/", OPENAI_API_KEY="sk-from-the-os")
