from datetime import datetime

import pytest

from bot_roulette.chat import ASSISTANT_OPENING_NUDGE, Assistant, Persona
from bot_roulette.persona import live_assistant_scenario

PASS = '{"passed": true}'


def test_the_task_is_made_the_persona_s_own_at_the_given_moment(make_llm):
    llm, fake = make_llm("- Tuesday, 21:15: you're on the sofa, the deadline is tomorrow")
    scene = live_assistant_scenario(
        llm, "CORE", "write a cover letter", when=datetime(2026, 11, 10, 21, 15)
    )
    prompt = fake.requests[0]["messages"][1]["content"]
    assert "Tuesday, 21:15, November 2026" in prompt
    assert "write a cover letter" in prompt and "CORE" in prompt
    assert scene.startswith("=== CURRENT SITUATION ===\n- Tuesday, 21:15")
    assert "[LEAVE]" in scene


def test_the_assistant_is_a_plain_model_with_its_own_history(make_llm):
    llm, fake = make_llm("Sure! Here's a draft.", RuntimeError("down"), "A shorter one.")
    assistant = Assistant(llm)
    assert assistant("cover letter pls") == "Sure! Here's a draft."
    with pytest.raises(RuntimeError):
        assistant("more")
    assert assistant("shorter") == "A shorter one."
    messages = fake.requests[2]["messages"]
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert [m["content"] for m in messages[1:]] == [
        "cover letter pls", "Sure! Here's a draft.", "shorter",
    ]


def test_only_the_assistant_searches_the_web(make_llm, persona_config, reply):
    llm, fake = make_llm(reply(text="cover letter pls"), PASS, "Sure!", "Here.")
    llm.web_search = {"tools": [{"type": "openrouter:web_search"}]}
    persona = Persona(llm, config_path=persona_config, scenario="ASSISTANT", scenario_text="SCENE")
    Assistant(llm)(persona(ASSISTANT_OPENING_NUDGE).reply)
    Assistant(llm, web_search=False)("cover letter pls")
    persona_turn, judge_turn, searching, not_searching = fake.requests
    assert "extra_body" not in persona_turn
    assert "tools" not in judge_turn.get("extra_body", {})
    assert searching["extra_body"] == {"tools": [{"type": "openrouter:web_search"}]}
    assert "extra_body" not in not_searching


def test_a_persona_using_the_assistant_knows_what_it_is_texting(make_llm, persona_config, reply):
    llm, fake = make_llm(reply(text="cover letter pls"), PASS)
    persona = Persona(llm, config_path=persona_config, scenario="ASSISTANT", scenario_text="SCENE")
    assert persona(ASSISTANT_OPENING_NUDGE).reply == "cover letter pls"
    assert "You are a real person\ntyping to an AI assistant app." in persona.history[0]["content"]
    judge_system, judge_user = fake.requests[1]["messages"]
    assert "a real person texting an AI assistant app" in judge_system["content"]
    assert "'them' is an AI assistant app" in judge_user["content"]


def test_a_stranger_chat_is_still_with_a_stranger(make_llm, persona_config, reply):
    llm, fake = make_llm(reply(), PASS)
    persona = Persona(llm, config_path=persona_config, scenario_text="SCENE")
    persona("ciao")
    assert (
        "You are a real person\nhaving a casual text conversation with someone you don't know well yet."
        in persona.history[0]["content"]
    )
    assert "a real person texting a stranger" in fake.requests[1]["messages"][0]["content"]


def test_with_an_app_the_rules_for_dealing_with_people_are_left_out(make_llm, persona_config):
    llm, _ = make_llm()
    prompts = {
        scenario: Persona(
            llm, config_path=persona_config, scenario=scenario, scenario_text="SCENE"
        ).history[0]["content"]
        for scenario in ("PAIRED", "ASSISTANT")
    }
    for person_only in (
        "tease the other", "=== HOW A REAL PERSON RESPONDS ===", "impression of the other person",
    ):
        assert person_only in prompts["PAIRED"] and person_only not in prompts["ASSISTANT"]
    assert "\n11. LANGUAGES." in prompts["PAIRED"] and "\n5. LANGUAGES." in prompts["ASSISTANT"]
