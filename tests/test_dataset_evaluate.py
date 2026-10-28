import json
import random

import pytest

from bot_roulette.chat import Conversation, run_conversation
from bot_roulette.evaluation import build_dataset, evaluate_dataset, load_dataset

RUBRIC = json.dumps({
    "scores": {"humanness": 4, "distinctiveness": 3, "consistency": 5,
               "engagement": 4, "no_assistant_leak": 5},
    "evidence": {"humanness": "messy"}, "verdict": "human", "summary": "fine",
})


@pytest.fixture
def pair(tmp_path, write_persona):
    root = tmp_path / "personas"
    a = write_persona(root, "anna_1", name="Anna", parent="anna", calibrated=True)
    b = write_persona(root, "rosa_1", name="Rosa", parent="rosa", calibrated=True)
    return root, a, b


def test_a_conversation_alternates_speakers(pair, make_llm, reply):
    _, a, b = pair
    llm, fake = make_llm("- story A", "- story B", reply(text="ciao [NEXT] tutto ok?"), reply(text="si"))
    conv = run_conversation(llm, a, b, conv_id="c1", max_turns=2, use_judge=False)
    assert [(t.who, t.bubbles) for t in conv.turns] == [("a", ["ciao", "tutto ok?"]), ("b", ["si"])]
    assert conv.ended == "turn_limit" and "story A" in conv.scenario_a
    # B is told exactly what A's bubbles said.
    assert fake.requests[3]["messages"][-1]["content"] == "ciao\ntutto ok?"


def test_dataset_then_evaluation(pair, make_llm, reply, tmp_path):
    root, a, b = pair
    out = tmp_path / "datasets" / "run"
    llm, _ = make_llm("- story A", "- story B", reply(text="ciao"))
    build_dataset(llm, [a, b], out_dir=out, max_turns=1, use_judge=False, rng=random.Random(0))
    assert [c.id for c in load_dataset(out)] == ["conv_0001"]

    metrics_only = evaluate_dataset(out, use_rubric=False)
    assert metrics_only["aggregate_metrics"]["turns"] == 1
    assert (out / "report.md").exists() and (out / "evaluation.json").exists()

    llm, fake = make_llm(RUBRIC)
    scored = evaluate_dataset(out, llm=llm, personas_dir=root)
    rubric = scored["conversations"][0]["rubric"]
    assert rubric["ok"] and rubric["verdict"] == "human"
    assert "Turin" in fake.requests[0]["messages"][1]["content"]  # profiles read from root


def test_the_rubric_needs_a_model_and_the_personas(pair, make_llm, reply, tmp_path):
    _, a, b = pair
    out = tmp_path / "run"
    llm, _ = make_llm("- story A", "- story B", reply())
    build_dataset(llm, [a, b], out_dir=out, max_turns=1, use_judge=False)
    with pytest.raises(ValueError, match="needs llm and personas_dir"):
        evaluate_dataset(out)


@pytest.mark.parametrize(
    "name_a, name_b, file_name",
    [
        ("Anna", "Rosa", "conv_0001_Anna__Rosa.json"),
        ("Anna Maria", 'Rosa/Bea: "?"', "conv_0001_Anna_Maria__Rosa_Bea.json"),
    ],
)
def test_conversation_file_names_are_safe_for_any_name(name_a, name_b, file_name):
    conv = Conversation(id="conv_0001", persona_a="a", persona_b="b", name_a=name_a,
                        name_b=name_b, moment="", scenario_a="", scenario_b="")
    assert conv.file_name() == file_name
