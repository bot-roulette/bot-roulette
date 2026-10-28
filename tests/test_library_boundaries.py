"""What the library promises as a library: no side effects on import,
quiet by default, its whole API importable from the top, and nothing
from the apps inside it."""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

PROBE = r"""
import json, os, sys
from types import SimpleNamespace as NS
before = dict(os.environ)
import bot_roulette
from bot_roulette import chat, creation, evaluation, llm, persona
from bot_roulette.llm import LLM, complete

# An empty answer then a good one: complete() logs a retry warning.
answers = iter(["", "ok"])
client = NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[
    NS(message=NS(content=next(answers), reasoning=None), finish_reason="stop")
]))))
assert complete(LLM(client, "m"), [], attempts=2) == "ok"
print(json.dumps({
    "env_changed": dict(os.environ) != before,
    "app_modules": sorted(m for m in ("fastapi", "uvicorn", "dotenv", "starlette") if m in sys.modules),
}))
"""


def test_import_reads_nothing_writes_nothing_and_logs_nothing(tmp_path):
    (tmp_path / ".env").write_text("MODEL=should-not-be-read\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("MODEL", "OPENROUTER_API_KEY")}
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report == {"env_changed": False, "app_modules": []}
    assert "asking again" not in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_the_package_has_no_entry_points_or_app_code():
    package = SRC / "bot_roulette"
    sources = {p.relative_to(package).as_posix(): p.read_text(encoding="utf-8") for p in package.rglob("*.py")}
    assert "chat/engine.py" in sources  # the subpackages are scanned too
    for name, text in sources.items():
        for marker in ("argparse", "__main__", "os.environ", "load_dotenv", "fastapi", "__file__"):
            assert marker not in text, f"{name} mentions {marker}"


def test_every_area_export_is_importable_from_the_package():
    import bot_roulette
    from bot_roulette import chat, creation, evaluation, llm, persona

    for area in (llm, persona, chat, creation, evaluation):
        missing = sorted(set(area.__all__) - set(bot_roulette.__all__))
        assert missing == [], f"{area.__name__} exports {missing}; bot_roulette doesn't"
    assert all(hasattr(bot_roulette, name) for name in bot_roulette.__all__)
