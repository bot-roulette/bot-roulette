"""Shared LLM plumbing: the model object, one call helper, JSON helpers.

Every model call in the package goes through ``complete(llm, ...)``,
where ``llm`` is an LLM: an OpenAI-compatible client plus the model to
ask. Nothing here reads the environment or any file: the application
builds the LLM (``LLM.openrouter(api_key, model)``, or
``LLM.openai_compatible(base_url, model)`` for any other OpenAI-compatible
server, a local one included) and hands it in.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from loguru import logger
from openai import OpenAI

from .openai_compatible import _check_url
from .openrouter import OPENROUTER_URL

# Total tries for one-shot generation steps (interviewer questions,
# analysis, enrichment, voice calibration, persona sketches, entry
# stories) when the model's output is unusable. One bad sample must not
# fail a long, paid creation job. Persona turns retry on their own (see
# chat.Persona) and the judge fails open, so both keep the default of 1.
GENERATION_ATTEMPTS = 3


class UnusableOutput(RuntimeError):
    """The model answered, but with nothing usable: empty content, only a
    reasoning trace, or output cut off at the length limit."""


class Cancelled(BaseException):
    """The user stopped the creation job running on this thread.

    A BaseException on purpose, like asyncio.CancelledError: the pipeline
    is full of fail-open ``except Exception`` handlers (the judge, voice
    calibration, dataset turns) that must not swallow a stop request and
    carry on as if nothing had happened.
    """


_job = threading.local()


def set_cancel_event(event: threading.Event | None) -> None:
    """Make every model call on THIS thread raise Cancelled once ``event``
    is set; None removes the hook. Only creation jobs install one (each
    runs on its own thread), so chat sessions are never affected."""
    _job.cancel_event = event


def check_cancelled() -> None:
    """Raise Cancelled if this thread's job was stopped. Called before
    every model request: a stop takes effect at the next call."""
    event = getattr(_job, "cancel_event", None)
    if event is not None and event.is_set():
        raise Cancelled("stopped by the user")


# The OpenAI client sends no request without a key: a server that needs
# none gets this placeholder, and ignores it.
_NO_KEY = "no-key"


@dataclass
class LLM:
    """The chat model every call goes to: an OpenAI-compatible client
    plus the model id to request.

    ``model`` may also be a zero-argument callable returning the model
    id, resolved before every call: a long-running program can switch
    models without rebuilding the objects that hold this LLM (the
    OpenAI client accepts a callable ``api_key`` for the same reason).

    ``json_format`` and ``reasoning_off`` are what a request carries to
    ask this server for a JSON object (``complete(as_json=True)``) and to
    skip the model's reasoning (``complete(reasoning=False)``). The
    defaults suit any OpenAI-compatible server: a JSON schema that allows
    any object (the older "json_object" mode is rejected by some, LM
    Studio for one), and nothing for reasoning, which most servers only
    switch off on the server itself. ``LLM.openrouter`` sets OpenRouter's.
    """

    client: OpenAI
    model: str | Callable[[], str]
    json_format: dict = field(default_factory=lambda: {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    })
    reasoning_off: dict = field(default_factory=dict)

    @classmethod
    def openrouter(
        cls, api_key: str | Callable[[], str], model: str | Callable[[], str]
    ) -> LLM:
        """An LLM served by OpenRouter (https://openrouter.ai)."""
        return cls(
            OpenAI(base_url=OPENROUTER_URL, api_key=api_key),
            model,
            json_format={"type": "json_object"},
            reasoning_off={"reasoning": {"effort": "none"}},
        )

    @classmethod
    def openai_compatible(
        cls,
        base_url: str,
        model: str | Callable[[], str],
        api_key: str | Callable[[], str] | None = None,
    ) -> LLM:
        """An LLM on any OpenAI-compatible server: a local one (Ollama, LM
        Studio, vLLM, llama.cpp...) or a hosted one. ``api_key`` only if
        the server wants one: without it, requests carry a placeholder,
        never a key the OpenAI client would pick up from the environment."""
        _check_url(base_url)

        def key() -> str:
            return (api_key() if callable(api_key) else api_key) or _NO_KEY

        return cls(OpenAI(base_url=base_url, api_key=key), model)

    def resolve_model(self) -> str:
        model = self.model() if callable(self.model) else self.model
        if not model:
            raise ValueError("no model set for this LLM")
        return model


def complete(
    llm: LLM,
    messages: list[dict],
    temperature: float = 1.0,
    *,
    as_json: bool = False,
    reasoning: bool = True,
    extra_body: dict | None = None,
    attempts: int = 1,
) -> str:
    """One chat completion. No max_tokens limit; raises UnusableOutput on
    empty, truncated or failed output.

    ``as_json`` asks for a JSON object and ``reasoning=False`` for no
    reasoning, each the way this LLM's server understands it (see LLM).
    ``extra_body`` adds any other fields to the request.

    ``message.reasoning`` is NEVER used as the answer. When a reasoning
    model leaves ``content`` empty it usually never reached an answer at
    all (it looped until the output limit), and its trace is not a
    deliverable: falling back to it once saved a 500 KB looping
    chain-of-thought as a persona's field notes, and the same path would
    send a trace to the user as chat bubbles. Raising instead lets every
    caller's retry / fail-open logic do its job. A trace some servers
    leave inline, as "<think>...</think>" before the answer, is cut off.

    ``attempts`` is how many times in total to ask when the output is
    unusable (see GENERATION_ATTEMPTS). That includes a failure OpenRouter
    reports DURING generation: it arrives as an HTTP 200, either with an
    "error" body and no choices or with a choice carrying an "error" next
    to whatever was written before it, which the OpenAI client neither
    raises on nor retries. Real HTTP errors are not retried here: the
    client already retries the transient ones itself. Every attempt goes
    to the server and model the call started with.
    """
    # Read once, together: the web app's settings can move ``llm`` to
    # another server while a call is under way, and a retry must not
    # pair one server's request options (or model) with the other.
    client, model = llm.client, llm.resolve_model()
    kwargs: dict = {}
    if as_json:
        kwargs["response_format"] = llm.json_format
    body = {**(extra_body or {}), **({} if reasoning else llm.reasoning_off)}
    if body:
        kwargs["extra_body"] = body
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        check_cancelled()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            **kwargs,
        )
        try:
            return _usable_content(first_choice(response))
        except UnusableOutput as e:
            if attempt == attempts:
                raise
            logger.warning(f"{e}; asking again ({attempt}/{attempts})")
    raise AssertionError("unreachable")


def first_choice(response):
    """The response's first choice, or UnusableOutput carrying the error.

    OpenRouter answers a failure that happens during generation (e.g. the
    upstream provider dying) with HTTP 200, an "error" object and no
    choices. Indexing ``response.choices`` then died with a bare TypeError
    or IndexError that hid OpenRouter's own message and skipped every
    retry.
    """
    choices = getattr(response, "choices", None)
    if choices:
        return choices[0]
    raise UnusableOutput(
        f"Model returned no choices ({_error_text(getattr(response, 'error', None))})"
    )


def _error_text(error) -> str:
    """The message of an "error" object OpenRouter sent (a dict, through
    the OpenAI client), or whatever else came."""
    if isinstance(error, dict):
        error = error.get("message") or error
    return str(error or "no error message")


# The parts of a list-shaped content that hold the model's reasoning.
_TRACE_PARTS = ("thinking", "reasoning")


def _split_reasoning(content: object) -> tuple[str, bool]:
    """(the answer, whether there was a trace) for content that may carry
    the model's reasoning inline, as some local servers return it:
    "<think>...</think>answer", or "...</think>answer" when the chat
    template opened the tag itself. A trace never closed has no answer.

    Some servers send a list of parts instead of a string (Mistral's
    reasoning models, for one): the answer is its "text" parts, and a
    "thinking" part is a trace. Content that is neither has no answer:
    str() would pass its repr, trace and all, for one."""
    traced = False
    if isinstance(content, list):
        parts = [part for part in content if isinstance(part, dict)]
        traced = any(part.get("type") in _TRACE_PARTS for part in parts)
        content = "".join(
            part["text"] for part in parts
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    elif not isinstance(content, str):
        return "", False
    if "</think>" in content:
        return content.rpartition("</think>")[2], True
    if content.lstrip().startswith("<think>"):
        return "", True
    return content, traced


def _usable_content(choice) -> str:
    finish_reason = getattr(choice, "finish_reason", None)
    error = getattr(choice, "error", None)
    if error or finish_reason == "error":
        # The other way OpenRouter reports a failure during generation
        # (see first_choice): a choice carrying an "error" next to
        # whatever was written before it, half an answer at best.
        raise UnusableOutput(f"Model generation failed ({_error_text(error)})")
    message = getattr(choice, "message", None)
    content, traced = _split_reasoning(getattr(message, "content", None))
    if not content.strip():
        if traced or getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None):
            raise UnusableOutput(
                "Model returned only a reasoning trace and no content"
            )
        raise UnusableOutput("Model returned empty content")
    if finish_reason in ("length", "content_filter"):
        # Cut off at the output limit or by the provider's content
        # filter: half an answer (or a runaway loop) must not pass for a
        # complete one.
        raise UnusableOutput(f"Model output was truncated (finish_reason={finish_reason})")
    return content.strip()


# A JSON string, a trailing comma or a Python literal, matched in one
# pass: a string is matched whole, so no repair reaches inside it.
_REPAIR_RE = re.compile(r'"(?:[^"\\]|\\.)*"|,(\s*[}\]])|\b(True|False|None)\b', re.DOTALL)
_JSON_LITERALS = {"True": "true", "False": "false", "None": "null"}


def _loads_object(blob: str) -> dict | None:
    """json.loads with a couple of repairs models actually emit."""
    for candidate in (blob, _repair_json(blob)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return data if isinstance(data, dict) else None
    return None


def _repair_json(blob: str) -> str:
    # Trailing commas, and Python literals instead of JSON ones. Both
    # show up when the model is "writing JSON" in a Python accent. Text
    # inside strings is left as written: a critique saying "None of it
    # is True" must not come back as "null of it is true".
    def repair(match: re.Match[str]) -> str:
        closing, literal = match.group(1, 2)
        if closing is not None:
            return closing  # what followed the trailing comma
        if literal is not None:
            return _JSON_LITERALS[literal]
        return match.group(0)  # a string
    return _REPAIR_RE.sub(repair, blob)


# How many "{" that never close extract_json looks past: each one costs
# another walk over the rest of the text.
_MAX_STRAY_BRACES = 10


def _scan_objects(text: str, begin: int, objects: list[dict]) -> int | None:
    """Add to ``objects`` every top-level object in ``text[begin:]`` that
    parses (see extract_json). Returns where an object still open at the
    end of the text starts, if one is."""
    start = None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[begin:], begin):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"' and depth:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    data = _loads_object(text[start : i + 1])
                    start = None
                    if data is not None:
                        objects.append(data)
    return start if depth else None


def extract_json(text: str) -> dict | None:
    """Pull a complete JSON object out of a model response.

    Both the judge and the evaluation rubric ask for "STRICT JSON, nothing
    else" and both get prose, markdown fences, trailing commentary or a
    second half-written object anyway. A greedy ``\\{.*\\}`` regex spans
    from the first brace to the LAST one in the whole response, so one
    stray "}" in trailing prose swallows garbage and json.loads dies on
    text that contained a perfectly good object.

    This walks the string instead, tracking brace depth (and ignoring
    braces inside strings / escapes), and collects every balanced object
    that actually parses. Strings are only tracked INSIDE an object: a
    lone quote in the prose before it ('her 5" phone') would otherwise
    open a string that never closes and hide the object entirely.
    Likewise a "{" that never closes, a stray brace in the prose or a
    half-written object, would swallow everything after it: the walk
    starts again right after it (up to _MAX_STRAY_BRACES times).
    Prefers an object that looks like a verdict (``passed``) or a rubric
    (``scores``) over an earlier dummy ``{...}`` the model mentioned in
    prose. Returns None when there is none.
    """
    if not text:
        return None
    objects: list[dict] = []
    begin = 0
    for _ in range(_MAX_STRAY_BRACES + 1):
        unclosed = _scan_objects(text, begin, objects)
        if unclosed is None:
            break
        begin = unclosed + 1
    if not objects:
        return None
    for key in ("passed", "scores"):
        for obj in objects:
            if key in obj:
                return obj
    return objects[0]
