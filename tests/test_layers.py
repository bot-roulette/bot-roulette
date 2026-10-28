"""The library's layers: each area imports only from the areas below it,
so the dependencies form a ladder and a cycle can't creep in (not even
through an import tucked inside a function)."""

import ast
from importlib.util import resolve_name
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "bot_roulette"

# Each area, and the areas it may import from. config.py counts as an
# area of its own; the package __init__ sits on top and re-exports them all.
ALLOWED = {
    "llm": set(),
    "config": set(),
    "persona": {"llm"},
    "chat": {"llm", "persona"},
    "creation": {"llm", "config", "persona", "chat"},
    "evaluation": {"llm", "persona", "chat"},
}


def imported_modules(source: str, package: str) -> list[str]:
    """Absolute names of the library modules ``source`` imports, with
    relative imports resolved against ``package``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            found.append(
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level else node.module or ""
            )
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names]
    return [m for m in found if m.split(".")[0] == "bot_roulette"]


def area_of(module: str) -> str | None:
    """bot_roulette.chat.judge -> chat; the package root -> None."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def test_relative_imports_are_resolved():
    source = (
        "from ..llm.core import LLM\nfrom .engine import Persona\n"
        "from . import parsing\nimport bot_roulette.persona\n"
        "def lazy():\n    from ..creation import enrich\n"
    )
    assert sorted(imported_modules(source, "bot_roulette.chat")) == [
        "bot_roulette.chat",
        "bot_roulette.chat.engine",
        "bot_roulette.creation",
        "bot_roulette.llm.core",
        "bot_roulette.persona",
    ]


def test_every_area_has_a_place_in_the_ladder():
    areas = {p.stem for p in PACKAGE.glob("*.py") if p.name != "__init__.py"}
    areas |= {p.name for p in PACKAGE.iterdir() if (p / "__init__.py").exists()}
    assert areas == set(ALLOWED)


def test_each_area_imports_only_from_the_areas_below_it():
    problems = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        area = area_of(".".join(parts).removesuffix(".__init__"))
        if area is None:
            continue
        for module in imported_modules(path.read_text(encoding="utf-8"), ".".join(parts[:-1])):
            target = area_of(module)
            if target != area and target not in ALLOWED[area]:
                problems.append(f"{path.relative_to(PACKAGE).as_posix()}: {area} -> {target or 'the package root'} ({module})")
    assert problems == []
