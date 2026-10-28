"""The rule that keeps the split honest: the CLI and the web app use only
the library's public API, and the library knows nothing about them.

The public API is what ``bot_roulette`` and its areas
(``bot_roulette.chat``, ``bot_roulette.llm``...) export: never a
private name, and never a module inside an area, so the library can
rearrange its modules without breaking the apps."""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "bot_roulette"
APP_SOURCES = sorted(
    [*(REPO / "cli" / "bot_roulette_cli").glob("*.py"),
     *(REPO / "webapp" / "bot_roulette_web").glob("*.py")]
)
# Every module and package of the library by dotted name. Compared as
# strings, case-sensitively: Judge is a class, judge a module (and a
# Windows file lookup would not tell them apart).
LIBRARY_MODULES = {
    ".".join(p.relative_to(PACKAGE.parent).with_suffix("").parts).removesuffix(".__init__")
    for p in PACKAGE.rglob("*.py")
}


def private_uses(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "bot_roulette":
            parts = node.module.split(".")
            found += [f"from {node.module}" for p in parts if p.startswith("_")]
            found += [f"{node.module}.{a.name}" for a in node.names if a.name.startswith("_")]
        elif isinstance(node, ast.Import):
            found += [
                a.name for a in node.names
                if a.name.split(".")[0] == "bot_roulette"
                and any(p.startswith("_") for p in a.name.split("."))
            ]
    found += re.findall(r"bot_roulette(?:\.\w+)*\._\w+", source)
    return found


def deep_imports(source: str) -> list[str]:
    """Library modules imported from deeper than ``bot_roulette.<area>``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append(node.module)
            # "from bot_roulette.chat import parsing" reaches in too.
            found += [f"{node.module}.{a.name}" for a in node.names]
    return [m for m in found if m in LIBRARY_MODULES and m.count(".") > 1]


def test_the_rule_catches_private_imports():
    assert private_uses("from bot_roulette.chat.parsing import _clean") == ["bot_roulette.chat.parsing._clean"]
    assert private_uses("from bot_roulette.llm import complete") == []


def test_the_rule_catches_imports_from_inside_an_area():
    assert deep_imports("from bot_roulette.chat.parsing import LEAVE_RE") == ["bot_roulette.chat.parsing"]
    assert deep_imports("from bot_roulette.chat import parsing") == ["bot_roulette.chat.parsing"]
    assert deep_imports("import bot_roulette.llm.openrouter") == ["bot_roulette.llm.openrouter"]
    assert deep_imports("from bot_roulette.chat import Judge, LEAVE_RE\nfrom bot_roulette import chat") == []


@pytest.mark.parametrize("path", APP_SOURCES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_apps_use_only_the_public_api(path):
    source = path.read_text(encoding="utf-8")
    assert private_uses(source) == []
    assert deep_imports(source) == []


def test_the_library_does_not_know_the_apps():
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "bot_roulette_cli" not in text and "bot_roulette_web" not in text, path.relative_to(PACKAGE)
