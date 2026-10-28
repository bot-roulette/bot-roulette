import pytest

from bot_roulette.persona import find_chat_ready_personas, find_enriched_personas, load_persona


def test_load_persona(persona_config):
    spec = load_persona(persona_config)
    # No chat_ready key (an older folder): calibrated implies ready.
    assert (spec.name, spec.full_name, spec.parent, spec.voice_calibrated, spec.chat_ready) == (
        "Giulia", "Giulia Rossi", "giulia", True, True,
    )
    assert "Turin" in spec.core
    assert "INTERVIEW" in spec.scenarios and "PAIRED" not in spec.scenarios


def test_finders_only_list_enriched_and_ready_personas(tmp_path, write_persona):
    write_persona(tmp_path, "anna", name="Anna")
    enriched = write_persona(tmp_path, "anna_20260101_000000", name="Anna", parent="anna", ready=False)
    ready = write_persona(tmp_path, "rosa_20260101_000000", name="Rosa", parent="rosa", calibrated=True)
    # Voice calibration skipped (no one to text): ready with its interview voice.
    lone = write_persona(tmp_path, "sara_20260101_000000", name="Sara", parent="sara", ready=True)
    assert find_enriched_personas(tmp_path) == [enriched, ready, lone]
    assert find_chat_ready_personas(tmp_path) == [ready, lone]


def test_load_persona_errors(tmp_path, write_persona):
    config = write_persona(tmp_path, "anna", name="Anna")
    (config.parent / "anna.md").unlink()
    with pytest.raises(FileNotFoundError):
        load_persona(config)
    config.write_text("name: Anna\n", encoding="utf-8")
    with pytest.raises(ValueError, match="core"):
        load_persona(config)
