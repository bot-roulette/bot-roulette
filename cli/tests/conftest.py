import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


class FakeClient:
    """Stands in for openai.OpenAI: scripted replies, in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.replies.pop(0), reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _persona(root: Path, folder: str, name: str, parent: str, calibrated: bool) -> Path:
    d = root / folder
    d.mkdir(parents=True)
    stem = name.lower()
    (d / f"{stem}.md").write_text(f"=== WHO YOU ARE ===\n{name}, 30.", encoding="utf-8")
    (d / f"{stem}.yaml").write_text(
        f"name: {name}\nfull_name: {name} Rossi\ncore: {stem}.md\n"
        f"parent: {parent}\nvoice_calibrated: {str(calibrated).lower()}\n",
        encoding="utf-8",
    )
    return d / f"{stem}.yaml"


@pytest.fixture(autouse=True)
def restore_environment():
    """The CLI loads .env into os.environ (override on): undo it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home folder: the shipped config.yaml and three small personas
    (Anna's v0.1 root, Anna enriched, Rosa enriched)."""
    shutil.copy(REPO / "config.yaml", tmp_path / "config.yaml")
    personas = tmp_path / "personas"
    monkeypatch.setenv("BOT_ROULETTE_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    return SimpleNamespace(
        path=tmp_path,
        anna_root=_persona(personas, "anna", "Anna", "none", False),
        anna=_persona(personas, "anna_1", "Anna", "anna", True),
        rosa=_persona(personas, "rosa_1", "Rosa", "rosa", True),
        FakeClient=FakeClient,
    )
