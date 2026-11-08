"""Image models: the portrait of a new persona (see creation/appearance.py).

An ImageModel is to pictures what LLM is to text: the application builds
it and hands it in, and nothing here reads the environment. It runs on
OpenRouter, through its image API (POST /images), or on any server that
follows OpenAI's image API (POST /images/generations), a local one
included.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .core import check_cancelled
from .openai_compatible import _check_url
from .openrouter import OPENROUTER_URL

# The media types an image comes back as, and the file extension of each.
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}

# The long side, in pixels, of an image asked of a server that takes a
# size: about the megapixel most open image models are made for.
_LONG_SIDE = 1024


@dataclass
class ImageModel:
    """The model pictures are asked of, and the API key of its server:
    either may be a zero-argument callable, resolved before every call,
    as for LLM. ``base_url`` is a server that follows OpenAI's image
    API, or None for OpenRouter."""

    api_key: str | Callable[[], str]
    model: str | Callable[[], str]
    base_url: str | None = None
    timeout: float = 300  # the slowest models on OpenRouter take a minute or two

    @classmethod
    def openrouter(
        cls, api_key: str | Callable[[], str], model: str | Callable[[], str]
    ) -> ImageModel:
        """An image model served by OpenRouter (https://openrouter.ai)."""
        return cls(api_key, model)

    @classmethod
    def openai_compatible(
        cls,
        base_url: str,
        model: str | Callable[[], str],
        api_key: str | Callable[[], str] | None = None,
    ) -> ImageModel:
        """An image model on any server that follows OpenAI's image API:
        a local one (LocalAI, stable-diffusion.cpp's sd-server...) or a
        hosted one. ``api_key`` only if the server wants one: without
        it, requests carry none."""
        _check_url(base_url)
        # A local server may load the model first, or draw on a CPU.
        return cls(api_key or "", model, base_url, timeout=600)


def _resolve(value: str | Callable[[], str]) -> str:
    return value() if callable(value) else value


def generate_image(
    image_model: ImageModel, prompt: str, *, aspect_ratio: str = "1:1"
) -> tuple[bytes, str]:
    """One image for ``prompt``: its bytes and its file extension.
    OpenRouter fits ``aspect_ratio`` to the ratios the model offers;
    another server is asked for that ratio in pixels (see _size).
    RuntimeError when no image comes back."""
    check_cancelled()
    request = {"model": _resolve(image_model.model), "prompt": prompt}
    if image_model.base_url is None:
        server, url = "OpenRouter", f"{OPENROUTER_URL}/images"
        request["aspect_ratio"] = aspect_ratio
    else:
        server = url = f"{image_model.base_url.rstrip('/')}/images/generations"
        # LocalAI, for one, answers with a link to the image unless asked for the bytes.
        request.update(size=_size(aspect_ratio), response_format="b64_json")
    api_key = _resolve(image_model.api_key)
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json=request,
        timeout=image_model.timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"{server} answered HTTP {response.status_code}: {response.text[:300]}"
        )
    images = response.json().get("data") or []
    if not images or not images[0].get("b64_json"):
        raise RuntimeError(f"{server} sent back no image")
    image = base64.b64decode(images[0]["b64_json"])
    return image, _extension(image, images[0].get("media_type"))


def _size(aspect_ratio: str) -> str:
    """``aspect_ratio`` ("3:4") in pixels ("768x1024"): the long side
    _LONG_SIDE, both sides multiples of 64, as image models want them."""
    width, height = (int(side) for side in aspect_ratio.split(":"))
    scale = _LONG_SIDE / max(width, height)
    return "x".join(str(round(side * scale / 64) * 64) for side in (width, height))


def _extension(image: bytes, media_type: str | None) -> str:
    """The file extension of ``image``: from the media type OpenRouter
    sends, else from its first bytes (OpenAI's API doesn't say)."""
    if media_type in _EXTENSIONS:
        return _EXTENSIONS[media_type]
    if image.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "webp"
    return "png"
