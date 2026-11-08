import base64

import httpx
import pytest

from bot_roulette.llm import ImageModel, generate_image, images, openai_compatible
from bot_roulette.llm.openai_compatible import (
    ServerUnavailable,
    check_server_model,
    is_served_model,
    server_models,
)

URL = "http://localhost:11434/v1"
MODELS = {"object": "list", "data": [{"id": "qwen2.5vl:7b"}, {"id": "gemma3:latest"}, {"object": "no id"}]}
PNG = b"\x89PNG\r\n\x1a\n pixels"


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


@pytest.fixture
def image_server(monkeypatch):
    """A server's POST /images/generations, faked: answers with
    state["image"], or an error when state["status"] says so; records
    the requests."""
    sent = []
    state = {"status": 200, "image": PNG}

    def fake_post(url, **kwargs):
        sent.append((url, kwargs))
        if state["status"] != 200:
            return httpx.Response(state["status"], text="no model loaded")
        image = base64.b64encode(state["image"]).decode()
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": image}]})

    monkeypatch.setattr(images.httpx, "post", fake_post)
    return sent, state


def test_an_image_server_is_asked_in_pixels_and_without_a_key(image_server):
    sent, _ = image_server
    model = ImageModel.openai_compatible(URL + "/", "z-image-turbo")
    assert generate_image(model, "a face", aspect_ratio="3:4") == (PNG, "png")
    ((url, request),) = sent
    assert url == URL + "/images/generations"
    assert request["headers"] == {}
    assert request["json"] == {
        "model": "z-image-turbo", "prompt": "a face", "size": "768x1024", "response_format": "b64_json",
    }


@pytest.mark.parametrize("aspect_ratio, size", [("1:1", "1024x1024"), ("16:9", "1024x576")])
def test_the_size_keeps_the_ratio(image_server, aspect_ratio, size):
    sent, _ = image_server
    generate_image(ImageModel.openai_compatible(URL, "m"), "a face", aspect_ratio=aspect_ratio)
    assert sent[0][1]["json"]["size"] == size


@pytest.mark.parametrize(
    "image, extension",
    [(b"\xff\xd8\xff\xe0 a jpeg", "jpg"), (b"RIFF\x10\x00\x00\x00WEBPVP8 ", "webp")],
)
def test_the_format_is_read_from_the_image_itself(image_server, image, extension):
    _, state = image_server
    state["image"] = image
    assert generate_image(ImageModel.openai_compatible(URL, "m"), "a face") == (image, extension)


def test_an_image_server_gets_its_own_key_and_is_named_when_it_fails(image_server):
    sent, state = image_server
    model = ImageModel.openai_compatible(URL, "m", api_key=lambda: "lm-key")
    generate_image(model, "a face")
    assert sent[-1][1]["headers"] == {"Authorization": "Bearer lm-key"}
    state["status"] = 500
    with pytest.raises(RuntimeError, match="/images/generations answered HTTP 500: no model loaded"):
        generate_image(model, "a face")
    with pytest.raises(ValueError, match="not an http"):
        ImageModel.openai_compatible("localhost:8888/v1", "m")
