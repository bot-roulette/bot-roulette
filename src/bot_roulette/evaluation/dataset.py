"""Offline dataset of persona-to-persona conversations.

``build_dataset`` has every pair of the given personas text each other
(``chat.run_conversation``: the web app's spectator mode, headless) and
writes the conversations to disk so they can be evaluated later (see
evaluate.py). The CLI's ``bot-roulette dataset`` is a front end to it.

Output layout::

    <out_dir>/
      manifest.json          run config + index of conversations
      conv_0001_Sara__Rosa.json
      conv_0002_...

Each conversation JSON holds the full turn list (state, thoughts, reply
bubbles, judge verdict) plus both entry stories. Note: the evaluator
currently reads only the profiles and the transcript (see
evaluate.score_conversation) — the stories are recorded for inspection
and for reproducing a conversation, not scored.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from ..chat.conversation import DEFAULT_MAX_TURNS, Conversation, Turn, run_conversation
from ..llm.core import LLM


def _pairs(configs: list[Path]) -> list[tuple[Path, Path]]:
    """All unordered pairs. Who speaks first is decided per repetition."""
    return list(itertools.combinations(configs, 2))


def build_dataset(
    llm: LLM,
    configs: list[Path],
    *,
    out_dir: Path,
    repeats: int = 1,
    max_turns: int = DEFAULT_MAX_TURNS,
    use_judge: bool = True,
    rng: random.Random | None = None,
) -> Path:
    """Run every pair `repeats` times and write the dataset to disk.

    Conversations are saved one file at a time, as they finish: a run of
    dozens of conversations is long enough that an interrupted batch must
    still leave usable data behind.
    """
    rng = rng or random.Random()
    pairs = _pairs(configs)
    if not pairs:
        raise ValueError(
            "need at least 2 enriched personas to build a dataset "
            f"(found {len(configs)})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(pairs) * repeats
    logger.info(
        f"dataset: {len(configs)} personas, {len(pairs)} pair(s) x {repeats} "
        f"= {total} conversation(s), up to {max_turns} turns each -> {out_dir}"
    )

    index: list[dict] = []
    created = datetime.now().isoformat(timespec="seconds")
    n = 0
    for rep in range(repeats):
        for cfg_a, cfg_b in pairs:
            n += 1
            # Alternate who opens across repetitions: the opener sets the
            # topic, so always starting with the same persona would bias
            # every conversation of that pair the same way.
            a, b = (cfg_a, cfg_b) if rep % 2 == 0 else (cfg_b, cfg_a)
            # Spread the moments over the week so entry stories don't all
            # land on the same hour of the same weekday.
            when = datetime.now() - timedelta(
                days=rng.randrange(7), hours=rng.randrange(24)
            )
            conv_id = f"conv_{n:04d}"
            logger.info(
                f"--- [{n}/{total}] {a.parent.name} x {b.parent.name} "
                f"(rep {rep + 1}) ---"
            )
            conv = run_conversation(
                llm,
                a,
                b,
                conv_id=conv_id,
                max_turns=max_turns,
                when=when,
                use_judge=use_judge,
            )
            path = out_dir / conv.file_name()
            path.write_text(
                json.dumps(asdict(conv), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            index.append(
                {
                    "id": conv.id,
                    "file": path.name,
                    "persona_a": conv.persona_a,
                    "persona_b": conv.persona_b,
                    "turns": len(conv.turns),
                    "ended": conv.ended,
                }
            )
            (out_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "created": created,
                        "personas": [c.parent.name for c in configs],
                        "repeats": repeats,
                        "max_turns": max_turns,
                        "judge": use_judge,
                        "conversations": index,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    logger.info(f"dataset written: {out_dir}")
    return out_dir


def load_conversation(path: str | Path) -> Conversation:
    """Read one conversation JSON back into a Conversation."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = [Turn(**t) for t in data.pop("turns", [])]
    return Conversation(**data, turns=turns)


def load_dataset(out_dir: str | Path) -> list[Conversation]:
    """All conversations of a run, in manifest order."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [out_dir / c["file"] for c in manifest.get("conversations", [])]
    else:
        # Tolerate a hand-assembled folder of conversation files.
        files = sorted(out_dir.glob("conv_*.json"))
    return [load_conversation(p) for p in files if p.exists()]
