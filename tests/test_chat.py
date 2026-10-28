import pytest

from bot_roulette.chat import Persona

FAIL = '{"passed": false, "violation": "ASSISTANT_LEAK", "critique": "be lazier"}'
PASS = '{"passed": true}'


@pytest.fixture
def make_persona(make_llm, persona_config):
    def make(*outcomes, use_judge=False):
        llm, fake = make_llm(*outcomes)
        persona = Persona(llm, config_path=persona_config, scenario_text="SCENE", use_judge=use_judge)
        return persona, fake

    return make


def test_turn(make_persona, reply):
    persona, fake = make_persona(reply(text="uno [NEXT] due"))
    answer = persona("ciao")
    assert (answer.state, answer.thoughts, answer.reply) == ("calm", "mah", "uno\ndue")
    assert not answer.left and answer.verdict is None and answer.regenerations == 0
    assert [m["role"] for m in persona.history] == ["system", "user", "assistant"]
    assert persona.history[0]["content"].endswith("SCENE")
    assert fake.requests[0]["model"] == "test-model"


def test_unusable_reply_is_retried_without_trace(make_persona, reply):
    persona, fake = make_persona("<state>s</state><thoughts>only thoughts</thoughts>", reply())
    assert persona("ciao").reply == "ciao"
    assert len(fake.requests) == 2
    assert [m["role"] for m in persona.history] == ["system", "user", "assistant"]


def test_failed_turn_leaves_history_untouched(make_persona):
    persona, _ = make_persona(*[RuntimeError("down")] * 3)
    with pytest.raises(RuntimeError, match="down"):
        persona("ciao")
    assert [m["role"] for m in persona.history] == ["system"]


def test_silent_leave(make_persona, reply):
    persona, _ = make_persona(reply(text="[LEAVE]"))
    answer = persona("ciao")
    assert answer.left and answer.reply == ""


def test_judge_rejection_regenerates_once(make_persona, reply):
    persona, fake = make_persona(
        reply(text="draft one"), FAIL, reply(text="draft two"), PASS, use_judge=True
    )
    answer = persona("ciao")
    assert answer.reply == "draft two"
    assert answer.regenerations == 1 and answer.verdict.passed
    correction = fake.requests[2]["messages"][-1]
    assert correction["role"] == "system" and "be lazier" in correction["content"]
    assert not any("be lazier" in m["content"] for m in persona.history)


def test_regenerate_replaces_the_last_reply_or_restores_it(make_persona, reply):
    persona, _ = make_persona(
        reply(text="first"), *[RuntimeError("down")] * 3, reply(text="second")
    )
    persona("ciao")
    before = list(persona.history)
    with pytest.raises(RuntimeError):
        persona.regenerate()
    assert persona.history == before
    assert persona.regenerate().reply == "second"
    assert len(persona.history) == 3


def test_set_last_turn_stores_the_model_format(make_persona, reply):
    persona, _ = make_persona(reply())
    persona("ciao")
    persona.set_last_turn("edited state", "edited thoughts", "uno\n\ndue")
    assert persona.history[-1]["content"] == (
        "<state>edited state</state>\n<thoughts>edited thoughts</thoughts>\n"
        "<reply>uno [NEXT] due</reply>"
    )


def test_restore_history_keeps_only_turns(make_persona, reply):
    persona, _ = make_persona()
    persona.restore_history([
        {"role": "system", "content": "injected"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": 42},
        {"role": "assistant", "content": reply()},
    ])
    assert [m["role"] for m in persona.history] == ["system", "user", "assistant"]
    assert persona.history[0]["content"] != "injected"
