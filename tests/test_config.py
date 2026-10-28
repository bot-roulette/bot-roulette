from pathlib import Path

import pytest
import yaml

from bot_roulette.config import Config, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def write(tmp_path, data) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def data():
    data = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    data.pop("web")  # the web app's own section: not the library's business
    return data


def test_the_library_sections_load(tmp_path, data):
    config = load_config(write(tmp_path, data))
    assert config.creation.interview_questions > 0
    assert config.random_personas.age_bands


def test_an_app_adds_its_sections_by_subclassing():
    class AppConfig(Config):
        web: dict

    config = load_config(REPO_CONFIG, schema=AppConfig)
    assert isinstance(config, AppConfig)
    assert config.web["judge"] is True


def test_sections_nobody_declared_are_rejected():
    with pytest.raises(RuntimeError, match="web"):
        load_config(REPO_CONFIG)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["creation"].update(intervew_questions=3), "intervew_questions"),
        (lambda d: d["creation"].pop("voice_turns"), "voice_turns"),
        (lambda d: d["creation"].update(voice_min_turns=99), "voice_min_turns"),
        # Names and countries are chosen when generating now: an older
        # config.yaml that still lists them is told which key to drop.
        (lambda d: d["random_personas"].update(countries=["Italy"]), "countries"),
        (lambda d: d["random_personas"]["age_bands"].append("25_34"), "25_34"),
    ],
)
def test_problems_are_named(tmp_path, data, mutate, message):
    mutate(data)
    with pytest.raises(RuntimeError, match=message):
        load_config(write(tmp_path, data))
