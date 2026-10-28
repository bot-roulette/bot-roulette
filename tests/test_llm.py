import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from bot_roulette.llm import LLM, Cancelled, UnusableOutput, complete, extract_json, set_cancel_event


@pytest.mark.parametrize(
    "text, expected",
    [
        ('```json\n{"passed": true}\n```', {"passed": True}),
        ('{"passed": False, "violation": "X",}', {"passed": False, "violation": "X"}),
        ('see {...} then {"scores": {"a": 3}}', {"scores": {"a": 3}}),
        ("nothing here", None),
        ('her 5" phone {"passed": true}', {"passed": True}),
        ('{"a": "brace } inside"}', {"a": "brace } inside"}),
        # A "{" that never closes doesn't hide what follows it.
        ('Use { for blocks. {"passed": false, "violation": "X"}', {"passed": False, "violation": "X"}),
        ('{"passed": tr\n{"passed": true}', {"passed": True}),
        # The repairs never reach inside strings.
        (
            '{"passed": False, "critique": "None of it is \\"True, ]\\"",}',
            {"passed": False, "critique": 'None of it is "True, ]"'},
        ),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_complete_sends_the_model_and_returns_stripped_content(make_llm):
    llm, fake = make_llm("  hello  ")
    assert complete(llm, [{"role": "user", "content": "hi"}], temperature=0.2) == "hello"
    request = fake.requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.2


def test_complete_asks_again_on_unusable_output(make_llm):
    llm, fake = make_llm("", {"content": "cut", "finish_reason": "length"}, "ok")
    assert complete(llm, [], attempts=3) == "ok"
    assert len(fake.requests) == 3


@pytest.mark.parametrize(
    "outcome, message",
    [
        ("   ", "empty content"),
        ({"content": "half", "finish_reason": "length"}, "truncated"),
        ({"content": "half", "finish_reason": "content_filter"}, "finish_reason=content_filter"),
        ({"content": "Ciao, sono Giu", "finish_reason": "error"}, "generation failed"),
        (SimpleNamespace(choices=None, error={"message": "upstream died"}), "upstream died"),
    ],
)
def test_complete_raises_unusable_output(make_llm, outcome, message):
    llm, _ = make_llm(outcome)
    with pytest.raises(UnusableOutput, match=message):
        complete(llm, [])


def test_every_attempt_goes_where_the_call_started(make_llm):
    llm, old = make_llm("", "ok")
    llm.json_format, llm.reasoning_off = {"type": "json_object"}, {"reasoning": {"effort": "none"}}
    _, new = make_llm("never sent")
    ask = old.chat.completions.create

    def ask_then_move(**request):
        # The settings panel moves the LLM to another server meanwhile.
        llm.client, llm.model, llm.json_format, llm.reasoning_off = new, "new/model", {"type": "text"}, {}
        return ask(**request)

    old.chat.completions.create = ask_then_move
    assert complete(llm, [], as_json=True, reasoning=False, attempts=2) == "ok"
    assert new.requests == []
    retry = old.requests[1]
    assert retry["model"] == "test-model"
    assert retry["response_format"] == {"type": "json_object"}
    assert retry["extra_body"] == {"reasoning": {"effort": "none"}}


def test_cancel_event_stops_before_the_request(make_llm):
    llm, fake = make_llm("never sent")
    event = threading.Event()
    event.set()
    set_cancel_event(event)
    try:
        with pytest.raises(Cancelled):
            complete(llm, [])
    finally:
        set_cancel_event(None)
    assert fake.requests == []


def test_the_model_can_be_a_provider_resolved_on_every_call(make_llm):
    llm, fake = make_llm("one", "two")
    current = {"model": "first/model"}
    llm.model = lambda: current["model"]
    complete(llm, [])
    current["model"] = "second/model"
    complete(llm, [])
    assert [r["model"] for r in fake.requests] == ["first/model", "second/model"]


def test_a_missing_model_fails_before_any_request(make_llm):
    llm, fake = make_llm("never sent")
    llm.model = ""
    with pytest.raises(ValueError, match="no model"):
        complete(llm, [])
    assert fake.requests == []


def test_openrouter_llm():
    keys = iter(["sk-one", "sk-two"])
    llm = LLM.openrouter(api_key=lambda: next(keys), model="a/b")
    assert str(llm.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert llm.resolve_model() == "a/b"
    assert llm.json_format == {"type": "json_object"}
    assert llm.reasoning_off == {"reasoning": {"effort": "none"}}


def _answering(seen, **choice):
    """An HTTP client whose every request gets a one-word chat completion
    (``choice`` overrides fields of its one choice)."""
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ciao"}, "finish_reason": "stop", **choice}],
        })
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_failure_reported_inside_the_choice_is_unusable():
    # OpenRouter's other shape of a failure during generation: HTTP 200,
    # and a choice carrying an "error" next to what was written before it.
    seen = []
    llm = LLM.openrouter(api_key="sk-or-test", model="a/b")
    llm.client = llm.client.with_options(http_client=_answering(
        seen, finish_reason="error", error={"code": 502, "message": "upstream died"},
    ))
    with pytest.raises(UnusableOutput, match="upstream died"):
        complete(llm, [], attempts=2)
    assert len(seen) == 2


@pytest.mark.parametrize("api_key, sent", [(None, "no-key"), ("k-123", "k-123"), (lambda: "", "no-key")])
def test_a_request_reaches_an_openai_compatible_server_with_its_own_key(monkeypatch, api_key, sent):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-os")  # never sent
    seen = []
    llm = LLM.openai_compatible("http://localhost:11434/v1", "qwen2.5vl:7b", api_key=api_key)
    llm.client = llm.client.with_options(http_client=_answering(seen))  # the real client, a fake network
    assert complete(llm, [{"role": "user", "content": "hi"}], as_json=True, reasoning=False) == "ciao"
    (request,) = seen
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {sent}"
    body = json.loads(request.content)
    assert body["model"] == "qwen2.5vl:7b"
    assert body["response_format"]["type"] == "json_schema"
    assert "reasoning" not in body


@pytest.mark.parametrize("url", [
    "localhost:11434/v1", "ftp://host/v1", "http://", " http://host/v1", "http://host/v1?x=1", "http://[::1/v1",
])
def test_a_server_address_must_be_http(url):
    with pytest.raises(ValueError, match="http"):
        LLM.openai_compatible(url, "m")


def test_json_and_no_reasoning_are_asked_the_way_the_server_takes_them(make_llm):
    llm, fake = make_llm("a", "b", "c")
    complete(llm, [])
    complete(llm, [], as_json=True, reasoning=False, extra_body={"top_k": 20})
    llm.json_format, llm.reasoning_off = {"type": "json_object"}, {"reasoning": {"effort": "none"}}
    complete(llm, [], as_json=True, reasoning=False)
    plain, any_server, openrouter_style = fake.requests
    assert "response_format" not in plain and "extra_body" not in plain
    assert any_server["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert any_server["extra_body"] == {"top_k": 20}
    assert openrouter_style["response_format"] == {"type": "json_object"}
    assert openrouter_style["extra_body"] == {"reasoning": {"effort": "none"}}


@pytest.mark.parametrize(
    "content, answer",
    [
        ("<think>Who's asking?</think>\n\nciao", "ciao"),
        ("the chat template opened the tag</think>ciao", "ciao"),
        ("<thoughts>kept</thoughts><reply>ciao</reply>", "<thoughts>kept</thoughts><reply>ciao</reply>"),
        # Content as a list of parts (Mistral's reasoning models).
        (
            [{"type": "thinking", "thinking": [{"type": "text", "text": "Who's asking?"}]}, {"type": "text", "text": "ciao"}],
            "ciao",
        ),
    ],
)
def test_a_reasoning_trace_left_in_the_answer_is_cut_off(make_llm, content, answer):
    llm, _ = make_llm({"content": content})
    assert complete(llm, []) == answer


def test_an_answer_that_is_all_reasoning_is_unusable(make_llm):
    thinking_only = [{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}]
    llm, _ = make_llm("<think>still thinking when it stopped", "<think>done</think>  ", {"content": thinking_only})
    for _ in range(3):
        with pytest.raises(UnusableOutput, match="only a reasoning trace"):
            complete(llm, [])
