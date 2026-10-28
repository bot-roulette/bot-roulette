from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bot_roulette.config import Config
from bot_roulette.llm import LLM

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeClient:
    """Stands in for openai.OpenAI: returns scripted completions in order
    and records every request.

    An outcome is the reply text (str), a dict with "content" and
    "finish_reason", an exception to raise, or a ready-made response
    object (anything else), returned as is.
    """

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, str):
            outcome = {"content": outcome}
        if isinstance(outcome, dict):
            message = SimpleNamespace(content=outcome.get("content"), reasoning=None)
            choice = SimpleNamespace(
                message=message, finish_reason=outcome.get("finish_reason", "stop")
            )
            return SimpleNamespace(choices=[choice])
        return outcome


@pytest.fixture
def make_llm():
    """make_llm(*outcomes) -> (the LLM to pass in, the fake client behind it)."""

    def factory(*outcomes):
        fake = FakeClient(*outcomes)
        return LLM(client=fake, model="test-model"), fake

    return factory


def _write_persona(root, folder, *, name="Giulia", parent="none", calibrated=False, ready=None):
    """A minimal persona folder; returns its YAML config path. ``ready``
    writes a chat_ready key (left out by default, like older folders)."""
    d = root / folder
    d.mkdir(parents=True)
    stem = name.lower()
    (d / f"{stem}.md").write_text(
        f"=== WHO YOU ARE ===\n{name}, 30, lives in Turin.\n"
        "Languages: Italian (native), English (B1).",
        encoding="utf-8",
    )
    (d / f"{stem}.yaml").write_text(
        f"name: {name}\nfull_name: {name} Rossi\ncore: {stem}.md\n"
        f"parent: {parent}\nvoice_calibrated: {str(calibrated).lower()}\n"
        + ("" if ready is None else f"chat_ready: {str(ready).lower()}\n"),
        encoding="utf-8",
    )
    return d / f"{stem}.yaml"


@pytest.fixture
def write_persona():
    return _write_persona


@pytest.fixture
def persona_config(tmp_path):
    return _write_persona(
        tmp_path, "giulia_20260101_000000", parent="giulia", calibrated=True
    )


@pytest.fixture
def reply():
    def make(state="calm", thoughts="mah", text="ciao"):
        return f"<state>{state}</state>\n<thoughts>{thoughts}</thoughts>\n<reply>{text}</reply>"

    return make


@pytest.fixture
def library_config():
    """The shipped config.yaml as the library sees it (no web section)."""
    data = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    data.pop("web")
    return Config.model_validate(data)
