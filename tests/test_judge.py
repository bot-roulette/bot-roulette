from bot_roulette.chat import Judge
from bot_roulette.persona import load_persona

HISTORY = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "translate this for me"},
]


def judge_with(make_llm, persona_config, *outcomes):
    llm, fake = make_llm(*outcomes)
    llm.reasoning_off = {"think": False}  # what this LLM's server would take
    return Judge(llm, load_persona(persona_config)), fake


def test_pass(make_llm, persona_config):
    judge, fake = judge_with(make_llm, persona_config, '{"passed": true}')
    assert judge(HISTORY, "s", "t", "ciao").passed
    request = fake.requests[0]
    assert request["temperature"] == 0.0
    # JSON and no reasoning, each asked the way the LLM's server takes it.
    assert request["response_format"] == judge.llm.json_format
    assert request["extra_body"] == {"think": False}


def test_fail_carries_violation_and_critique(make_llm, persona_config):
    judge, _ = judge_with(
        make_llm, persona_config,
        '{"passed": false, "violation": "assistant_leak", "critique": "be lazier"}',
    )
    verdict = judge(HISTORY, "s", "t", "here is the full translation")
    assert (verdict.passed, verdict.violation, verdict.critique) == (
        False, "ASSISTANT_LEAK", "be lazier",
    )


def test_garbage_and_errors_fail_open(make_llm, persona_config):
    judge, _ = judge_with(make_llm, persona_config, "no json at all", RuntimeError("API down"))
    assert judge(HISTORY, "s", "t", "ciao").passed
    assert judge(HISTORY, "s", "t", "ciao").passed


def test_history_shows_only_what_was_sent(make_llm, persona_config):
    judge, fake = judge_with(make_llm, persona_config, '{"passed": true}')
    history = HISTORY + [
        {"role": "assistant", "content": "<state>s</state><thoughts>SECRET</thoughts><reply>uno [NEXT] due</reply>"},
        {"role": "user", "content": "e poi?"},
    ]
    judge(history, "s", "t", "boh")
    prompt = fake.requests[0]["messages"][1]["content"]
    assert "Giulia: uno due" in prompt
    assert "SECRET" not in prompt
