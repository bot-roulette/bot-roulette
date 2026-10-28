import base64
import io
import json
import random

import pytest
import yaml
from PIL import Image

from bot_roulette.chat import Persona
from bot_roulette.chat.conversation import Conversation, Turn
from bot_roulette.creation import (
    PersonaIdentity,
    build_random_persona,
    develop_persona,
    resume_development,
)
from bot_roulette.creation.cv_parser import create_root_persona, write_root_persona
from bot_roulette.creation.enrich import (
    _extract_how_you_text,
    _pick_voice_partners,
    _replace_how_you_text,
    calibrate_voice,
)
from bot_roulette.creation.interviewer import END_MARKER, Interviewer
from bot_roulette.creation.random_persona import (
    generate_random_persona,
    sample_seed_traits,
    validate_fixed_seed_traits,
)
from bot_roulette.llm import UnusableOutput
from bot_roulette.persona import load_persona

ENRICHED_CORE = (
    "=== WHO YOU ARE ===\nEnriched.\n=== YOUR PEOPLE ===\nSome.\n"
    "=== MEMORIES & STORIES ===\nScenes.\n=== HOW YOU TEXT ===\nShort."
)

ANNA = PersonaIdentity(first_name="Anna", surname="Rossi", sex="female", country="Italy")


def test_root_personas_are_written_into_the_given_folder(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    first = write_root_persona("Anna Maria!", "Anna Maria Rossi", "core", root=root)
    second = write_root_persona("Anna Maria!", "Anna Maria Rossi", "core", root=root)
    assert first == root.resolve() / "anna_maria" / "anna_maria.yaml"
    assert second.parent.name == "anna_maria2"
    assert load_persona(first).parent == "none"


def make_pdf(*pages):
    """A real PDF; each page is (text or None for an empty page, width,
    height), in points."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text, width, height in pages:
        number = len(objects) + 1
        kids.append(f"{number} 0 R")
        page = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}]"
        if text is None:
            objects.append(f"{page} >>".encode())
            continue
        stream = f"BT /F1 24 Tf 72 {height - 100} Td ({text}) Tj ET".encode()
        objects.append(f"{page} /Contents {number + 1} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    pdf, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    return pdf + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)


def test_a_cv_is_read_page_by_page_with_the_model_reasoning_off(tmp_path, make_llm):
    pdf = tmp_path / "cv.pdf"
    pdf.write_bytes(make_pdf(
        ("Anna Rossi", 595, 842),
        (None, 595, 842),             # nothing on it: never sent
        ("Nurse", 5000, 7000),        # poster-sized: rendered smaller
        ("Page 4", 595, 842),
    ))
    llm, fake = make_llm(
        "# Anna Rossi",
        "Nurse in Turin since 2019.",
        "**[Blank page]**",
        "NAME: Anna | FULL_NAME: Anna Rossi\n\nAnna is a nurse in Turin. Languages: Italian.",
    )
    llm.reasoning_off = {"think": False}  # what this LLM's server would take
    config = create_root_persona(llm, pdf, root=tmp_path / "personas")
    assert load_persona(config).full_name == "Anna Rossi"
    *reads, extraction = fake.requests
    sizes = []
    for number, read in enumerate(reads, 1):
        (message,) = read["messages"]
        prompt, image = message["content"]
        assert f"page {number} of 3 " in prompt["text"]
        url = image["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        sizes.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size)
    assert sizes == [(1190, 1684), (1429, 2000), (1190, 1684)]
    # A page the model calls blank is left out of what the extractor reads.
    assert extraction["messages"][-1]["content"] == "# Anna Rossi\n\nNurse in Turin since 2019."
    for request in (*reads, extraction):
        assert request["temperature"] == 0
        assert request["extra_body"] == {"think": False}


def test_a_document_without_text_is_rejected_before_the_extraction(tmp_path, make_llm):
    pdf = tmp_path / "cv.pdf"
    pdf.write_bytes(make_pdf((None, 595, 842), ("a logo", 595, 842)))
    llm, fake = make_llm("[blank page]")
    with pytest.raises(UnusableOutput, match="no text"):
        create_root_persona(llm, pdf, root=tmp_path / "personas")
    assert len(fake.requests) == 1
    assert not (tmp_path / "personas").exists()


def test_the_interview_opens_with_a_cue_kept_out_of_the_transcript(tmp_path, make_llm, write_persona, reply):
    config = write_persona(tmp_path, "anna", name="Anna")
    llm, fake = make_llm(
        "Tell me about your life.",
        reply(text="nata a Torino"),
        f"Thank you. {END_MARKER}",
    )
    persona = Persona(llm, config_path=config, scenario="INTERVIEW", use_judge=False)
    transcript = Interviewer(llm, max_turns=5)(persona)
    # The conversation, from the first request on, opens with a user turn.
    assert fake.requests[0]["messages"][1] == {"role": "user", "content": Interviewer.OPENING_CUE}
    assert transcript == [
        {"role": "interviewer", "content": "Tell me about your life."},
        {"role": "respondent", "content": "nata a Torino"},
        {"role": "interviewer", "content": "Thank you."},
    ]


def test_seed_traits_keep_the_identity_and_draw_the_background(library_config):
    lists = library_config.random_personas
    a = sample_seed_traits(random.Random(7), lists, ANNA)
    assert a == sample_seed_traits(random.Random(7), lists, ANNA)
    assert (a["first_name"], a["surname"], a["sex"], a["country"]) == ("Anna", "Rossi", "female", "Italy")
    assert a["age_band"] in lists.age_bands and a["education"] in lists.education
    assert "chat_language" not in a


def test_fixed_seed_traits_are_kept_and_the_rest_stays_valid(library_config):
    lists = library_config.random_personas
    identity = PersonaIdentity(
        first_name="Kenji", surname="Sato", sex="male", country="Japan", chat_language="English"
    )
    fixed = {
        "age_band": lists.age_bands[0],
        "work_area": "tech",
        "employment_stability": lists.employment_stability[0],
    }
    traits = sample_seed_traits(random.Random(7), lists, identity, fixed)
    assert all(traits[key] == value for key, value in fixed.items())
    assert traits["chat_language"] == "English"
    assert traits["life_event"] in lists.life_events


def test_a_fixed_retirement_draws_a_retirement_age(library_config):
    lists = library_config.random_personas
    for seed in range(20):
        traits = sample_seed_traits(random.Random(seed), lists, ANNA, {"work_area": "retired"})
        assert int(traits["age_band"].split("-")[0]) >= 55
        assert "employment_stability" not in traits


@pytest.mark.parametrize(
    "age_bands, fixed",
    [
        (None, {"age_band": "25-34", "work_area": "retired"}),
        # No age band in config.yaml allows retirement at all.
        (["25-34"], {"work_area": "retired"}),
    ],
)
def test_impossible_fixed_seed_traits_are_rejected(library_config, age_bands, fixed):
    lists = library_config.random_personas
    if age_bands:
        lists = lists.model_copy(update={"age_bands": age_bands})
    with pytest.raises(ValueError, match="retired requires"):
        sample_seed_traits(random.Random(7), lists, ANNA, fixed)


def test_fixed_seed_traits_can_be_in_your_own_words(library_config):
    lists = library_config.random_personas
    fixed = {
        "age_band": " 40 – 45 ",
        "work_area": " night-shift nurse in a\nchildren's hospital ",
        "settlement": "Bergamo",
    }
    traits = sample_seed_traits(random.Random(7), lists, ANNA, fixed)
    assert (traits["age_band"], traits["work_area"], traits["settlement"]) == (
        "40-45", "night-shift nurse in a children's hospital", "Bergamo",
    )
    # Not an area of work the guards know: a stability is drawn as usual.
    assert traits["employment_stability"] in lists.employment_stability


def test_traits_in_your_own_words_reach_the_generator_as_written(make_llm, library_config):
    llm, fake = make_llm("NAME: Anna | FULL_NAME: Anna Rossi\n\nAnna Rossi is 37. Languages: Italian.")
    result = generate_random_persona(
        llm, ANNA, library_config.random_personas, seed=7,
        fixed_traits={"age_band": "37", "settlement": "Città Alta, Bergamo"},
    )
    assert result["traits"]["age_band"] == "37"
    prompt = fake.requests[0]["messages"][-1]["content"]
    assert "- age band: 37\n" in prompt and "- settlement: Città Alta, Bergamo\n" in prompt


@pytest.mark.parametrize(
    "fixed, problem",
    [
        ({"age_band": "17"}, "invalid age_band"),
        ({"age_band": "45-40"}, "invalid age_band"),
        ({"age_band": "70-101"}, "invalid age_band"),
        ({"age_band": "late thirties"}, "invalid age_band"),
        ({"settlement": "  "}, "can't be empty"),
        ({"settlement": "Berga\x00mo"}, "plain text"),
        ({"work_area": "x" * 151}, "at most 150"),
        # A listed value stays one however it's spaced, and keeps its guards.
        ({"work_area": " retired ", "age_band": "50"}, "retired requires"),
        ({"work_area": "student with side jobs", "employment_stability": "a permanent post"}, "precarious"),
    ],
)
def test_bad_fixed_seed_traits_are_rejected(library_config, fixed, problem):
    with pytest.raises(ValueError, match=problem):
        validate_fixed_seed_traits(fixed, library_config.random_personas)


def test_an_identity_is_one_line_of_plain_text():
    identity = PersonaIdentity(
        first_name=" Anna\nMaria ", surname="De  Luca", sex="female",
        country="Italy", chat_language="  ",
    )
    assert (identity.first_name, identity.surname) == ("Anna Maria", "De Luca")
    assert identity.chat_language is None


@pytest.mark.parametrize(
    "changes, problem",
    [
        ({"first_name": "  "}, "can't be empty"),
        ({"surname": "Ros\x00si"}, "plain text"),
        ({"country": "x" * 61}, "at most 60"),
        ({"sex": "other"}, "female"),
    ],
)
def test_a_bad_identity_is_rejected(changes, problem):
    with pytest.raises(ValueError, match=problem):
        PersonaIdentity(**{**ANNA.model_dump(), **changes})


def test_random_persona_runs_the_whole_pipeline_inside_root(tmp_path, make_llm, library_config):
    root = tmp_path / "personas"
    llm, fake = make_llm(
        "NAME: Zelda | FULL_NAME: Zelda Moon\n\nZelda Moon is 30. Languages: Italian.",
        f"Tell me about your life. {END_MARKER}",
        "FIELD NOTES",
        ENRICHED_CORE,
    )
    enriched = build_random_persona(
        llm, identity=ANNA, root=root, creation=library_config.creation,
        random_personas=library_config.random_personas, seed=7,
    )
    traits = sample_seed_traits(random.Random(7), library_config.random_personas, ANNA)
    spec = load_persona(enriched)
    assert enriched.parent.parent == root.resolve()
    assert spec.parent == "anna" and not spec.voice_calibrated and spec.core == ENRICHED_CORE
    # Nobody to calibrate against: the first persona can chat anyway.
    assert spec.chat_ready
    assert spec.name == "Anna"
    assert "- first name: Anna\n" in fake.requests[0]["messages"][-1]["content"]
    # The model's own name for the person never survives.
    assert (root / "anna" / "anna.md").read_text(encoding="utf-8").startswith("Anna Rossi is 30.")
    root_cfg = yaml.safe_load((root / "anna" / "anna.yaml").read_text(encoding="utf-8"))
    assert root_cfg["seed_traits"] == traits
    assert list((root / "anna").glob("interview_*.json"))
    # No partner exists yet, so voice calibration made no call.
    assert len(fake.requests) == 4
    assert {r["model"] for r in fake.requests} == {"test-model"}


def test_a_saved_interview_is_reused(tmp_path, make_llm, library_config, write_persona):
    config = write_persona(tmp_path, "anna", name="Anna")
    (config.parent / "interview_anna_1.json").write_text(
        json.dumps([{"role": "interviewer", "content": "q"}, {"role": "respondent", "content": "a"}]),
        encoding="utf-8",
    )
    llm, fake = make_llm("FIELD NOTES", ENRICHED_CORE)
    enriched = develop_persona(llm, config, creation=library_config.creation)
    assert load_persona(enriched).parent == "anna"
    assert len(fake.requests) == 2


def test_voice_partners_come_from_the_same_personas_folder(tmp_path, write_persona):
    root, elsewhere = tmp_path / "personas", tmp_path / "elsewhere"
    me = write_persona(root, "anna_1", name="Anna", parent="anna")
    partner = write_persona(root, "rosa_1", name="Rosa", parent="rosa", calibrated=True)
    write_persona(root, "anna_2", name="Anna", parent="anna")  # a rebuild of me
    write_persona(root, "bea", name="Bea")  # a v0.1 root
    write_persona(elsewhere, "carla_1", name="Carla", parent="carla", calibrated=True)
    assert _pick_voice_partners(me, 2) == [partner]


NEW_VOICE = "=== HOW YOU TEXT ===\nshort lowercase bubbles, rarely asks back.\n"


@pytest.mark.parametrize(
    "how_you_text, after",
    [
        ("=== HOW YOU TEXT ===\nold stub.\n\n", "=== YOUR PEOPLE ===\n- Marco, her brother.\n"),
        # An empty section must not swallow the section after it.
        ("=== HOW YOU TEXT ===\n\n", "=== YOUR PEOPLE ===\n- Marco, her brother.\n"),
        ("=== HOW YOU TEXT ===\nold stub.\n", ""),
        ("=== HOW YOU TEXT ===", ""),
    ],
)
def test_voice_calibration_replaces_only_how_you_text(how_you_text, after):
    core = "=== WHO YOU ARE ===\nAnna.\n\n" + how_you_text + after
    assert _extract_how_you_text(core).startswith("=== HOW YOU TEXT ===")
    assert "YOUR PEOPLE" not in _extract_how_you_text(core)
    updated = _replace_how_you_text(core, NEW_VOICE)
    assert updated.count("=== HOW YOU TEXT ===") == 1
    assert "short lowercase bubbles" in updated and "old stub" not in updated
    assert updated.startswith("=== WHO YOU ARE ===\nAnna.\n\n=== HOW YOU TEXT ===")
    assert after.strip() in updated


def _stranger_chat(turns):
    """Stands in for run_conversation: a calibration chat of ``turns`` replies."""

    def run(llm, config_a, config_b, *, conv_id, **_):
        return Conversation(
            id=conv_id, persona_a=config_a.parent.name, persona_b=config_b.parent.name,
            name_a="A", name_b="B", moment="", scenario_a="", scenario_b="",
            turns=[
                Turn(index=i, who="ab"[i % 2], name="AB"[i % 2], state="", thoughts="", bubbles=["ciao"])
                for i in range(turns)
            ],
            ended="turn_limit",
        )

    return run


@pytest.mark.parametrize("rewritten", [False, True])
def test_with_a_partner_only_a_finished_calibration_makes_a_persona_chat_ready(
    tmp_path, make_llm, library_config, write_persona, monkeypatch, rewritten
):
    creation = library_config.creation
    me = write_persona(tmp_path, "anna_1", name="Anna", parent="anna", ready=False)
    write_persona(tmp_path, "rosa_1", name="Rosa", parent="rosa", calibrated=True)
    turns = creation.voice_min_turns if rewritten else creation.voice_min_turns - 1
    monkeypatch.setattr("bot_roulette.creation.enrich.run_conversation", _stranger_chat(turns))
    # Chats too short to use never reach the calibrator.
    llm, fake = make_llm(*([NEW_VOICE] if rewritten else []))
    assert calibrate_voice(llm, me, creation=creation) is rewritten
    spec = load_persona(me)
    assert (spec.voice_calibrated, spec.chat_ready) == (rewritten, rewritten)
    assert ("short lowercase bubbles" in spec.core) is rewritten
    assert len(fake.requests) == int(rewritten)


def test_a_developed_root_is_not_developed_twice(tmp_path, make_llm, library_config, write_persona):
    root = write_persona(tmp_path, "anna", name="Anna")
    write_persona(tmp_path, "anna_1", name="Anna", parent="anna")
    llm, fake = make_llm()
    with pytest.raises(ValueError, match="already developed"):
        resume_development(llm, root, creation=library_config.creation)
    assert fake.requests == []
