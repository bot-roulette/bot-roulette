"""Quality evaluation of a conversation dataset (see dataset.py).

Two layers, deliberately separate:

1. MECHANICAL metrics, computed in Python — no LLM, no opinion: turn
   counts, bubble lengths, who spoke more, judge rejections, format
   leaks (a "<" surviving into a bubble, a stray [NEXT]/[LEAVE] marker,
   an empty reply). These are cheap regression signals: if a prompt
   change makes the model emit tags again, this catches it without
   asking a model whether it liked the conversation.
2. An LLM RUBRIC pass per conversation: 1-5 scores with a one-line
   justification per dimension, plus a verdict on whether the exchange
   reads as two humans texting.

The rubric is the opposite of judge.py by design. The judge is a
fail-open gate on a single response, biased hard toward PASS, and only
looks for simulation-breaking violations. This is an offline critic on a
whole conversation, and it is asked to be demanding: nobody is waiting
for the answer and an inflated score is worthless.

``evaluate_dataset`` evaluates a dataset folder written by
dataset.build_dataset (the CLI's ``bot-roulette evaluate`` is a front
end to it).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from ..chat.conversation import Conversation
from ..llm.core import LLM, complete, extract_json
from ..persona.spec import load_persona
from .dataset import load_dataset

# The rubric dimensions. Kept small and orthogonal: more dimensions
# would just correlate with each other and dilute the signal. Each is
# scored 1-5 (1 = clearly broken, 5 = indistinguishable from real).
DIMENSIONS: dict[str, str] = {
    "humanness": (
        "Would a reader believe two real people typed this? Judge rhythm, "
        "typos, register, message length, how questions are answered or "
        "dodged. Symmetrical ping-pong of tidy questions-and-answers is a "
        "LOW score: real texting is lopsided and messy."
    ),
    "distinctiveness": (
        "Are the two people clearly DIFFERENT from each other, in voice, "
        "vocabulary, interests and attitude? If you could swap their "
        "messages without anyone noticing, score 1-2. If their profiles "
        "are audible in how they write, score high."
    ),
    "consistency": (
        "Does each person stay the same person, and stay consistent with "
        "their profile and with what they themselves said earlier? Count "
        "contradictions of concrete facts, drifting personality, forgotten "
        "statements."
    ),
    "engagement": (
        "Does the conversation go somewhere — topics developing, "
        "curiosity, friction, jokes, a reason to keep texting? A polite "
        "exchange of nothing is a LOW score, even if flawless."
    ),
    "no_assistant_leak": (
        "Is either side behaving like an AI assistant — being unusually "
        "helpful, thorough, accurate, structured, eager to be of service, "
        "or performing tasks well? 5 = no trace of it. 1 = one of them is "
        "obviously a chatbot."
    ),
}

_SCALE = """Scoring scale (be strict; 3 is an ordinary, unremarkable result):
5 = indistinguishable from a real human exchange
4 = convincing, with a small blemish
3 = mixed; a careful reader would suspect something
2 = clearly artificial in places
1 = broken"""

_RUBRIC_SYSTEM = """You are a demanding evaluator of simulated-person conversations. Two
language models were each given the profile of an ordinary person and
made to text each other as strangers paired by a chat app. You receive
both profiles and the full transcript, and you rate the CONVERSATION.

You are NOT a safety or politeness reviewer. Rudeness, boredom, silence,
wrong facts about the world, refusals, dead ends and abrupt endings are
all realistic and must NOT be penalized — a real stranger chat is often
dull and unhelpful. Penalize ARTIFICIALITY: symmetry, tidiness,
eagerness, essay-length messages, therapy-speak, interviewing each other,
uncanny recall, and any hint of an assistant trying to be useful.

{scale}

Rate each dimension:
{dimensions}

Answer with STRICT JSON, nothing else:
{{"scores": {{{score_keys}}},
 "evidence": {{"<dimension>": "<one short sentence quoting or pointing at what decided the score>", ...}},
 "verdict": "human" | "suspicious" | "artificial",
 "summary": "<two sentences max: the single biggest strength and the single biggest problem>"}}

Every dimension must appear in both "scores" (integer 1-5) and
"evidence". Do not add commentary outside the JSON."""


_RUBRIC_USER = """=== PROFILE: {name_a} ===
{core_a}

=== PROFILE: {name_b} ===
{core_b}

=== HOW THE CONVERSATION ENDED ===
{ended}

=== TRANSCRIPT ({name_a} opened) ===
{transcript}

Rate this conversation. STRICT JSON only."""


# Format leaks the mechanical pass looks for in delivered bubbles. All of
# these are supposed to be impossible: they mean the parser in parsing.py
# let something through.
# Imported from parsing.py rather than re-declared: this detector must
# track the PARSER, so a copy would silently drift (it did — the local
# copy kept the old broad attribute pattern and flagged ordinary prose
# like "a < b ma 5 > 3" as a leak while the parser had already stopped
# stripping it, producing false parser-regression alarms).
from ..chat.parsing import STRAY_TAGS_RE as _TAG_LEAK_RE

_MARKER_LEAK_RE = re.compile(r"\[\s*(?:next|leave)\s*\]", re.IGNORECASE)

# Decoding artifacts: characters outside Latin/punctuation/symbol ranges,
# i.e. CJK, Cyrillic, Arabic, Devanagari... A Chinese-trained model under
# multilingual pressure occasionally drops a token from its dominant
# script mid-sentence ("sono già功能性 sveglia"). This is a HARDER
# immersion break than anything judge.py looks for — no human types that
# — and it is invisible to the tag/marker checks, so it gets its own
# counter. Latin-1/Latin Extended (accents), general punctuation,
# currency symbols and emoji are all legitimate and excluded — including
# the emoji that live in older symbol blocks (™ ⌛ ⏰ ▶ ⭐ ⬆ ①), which
# are not a script and must not count as artifacts.
_SCRIPT_ARTIFACT_RE = re.compile(
    r"[^"
    r"\u0000-\u024f"      # ASCII + Latin-1 + Latin Extended-A/B
    r"\u0250-\u02af"      # IPA extensions (rare but harmless)
    r"\u0300-\u036f"      # combining diacritics
    r"\u2000-\u206f"      # general punctuation (curly quotes, dashes, …)
    r"\u20a0-\u20bf"      # currency symbols
    r"\u20e3"             # combining enclosing keycap (1️⃣)
    r"\u2100-\u218f"      # letterlike symbols, number forms (™, ℹ, ½)
    r"\u2190-\u21ff"      # arrows
    r"\u2300-\u23ff"      # misc technical (⌚, ⌛, ⏰, ⏩)
    r"\u2460-\u24ff"      # enclosed alphanumerics (①, Ⓜ)
    r"\u2500-\u25ff"      # box drawing, geometric shapes (▶, ◀, ■)
    r"\u2600-\u27bf"      # dingbats / misc symbols (emoji-ish)
    r"\u2900-\u297f"      # supplemental arrows (⤴, ⤵)
    r"\u2b00-\u2bff"      # misc symbols and arrows (⭐, ⬆, ⭕)
    r"\u3030\u303d\u3297\u3299"  # the only emoji in CJK blocks (〰 〽 ㊗ ㊙)
    r"\U0001f000-\U0001faff"  # emoji planes
    r"\ufe0f\u200d"       # variation selector + ZWJ (emoji sequences)
    r"\U000e0020-\U000e007f"  # tag characters (subdivision flags)
    r"]"
)

# Immediate self-repetition inside ONE turn: the same 4-word sequence
# twice. Real people repeat themselves across a conversation, but a
# single message containing "e tu come ti chiami" twice is a decoding
# artifact. Measured on the first real dataset: 3 turns out of 75.
_WORD_RE = re.compile(r"[\w'\u00e0-\u00ff]+")
_DUP_NGRAM = 4


@dataclass
class Rubric:
    """One LLM rubric result. `ok` is False when the pass itself failed."""

    ok: bool = True
    scores: dict[str, int] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    verdict: str = ""
    summary: str = ""
    error: str = ""

    @property
    def mean(self) -> float | None:
        return statistics.mean(self.scores.values()) if self.scores else None


def _format_prompt() -> str:
    dimensions = "\n".join(
        f"- {key}: {desc}" for key, desc in DIMENSIONS.items()
    )
    score_keys = ", ".join(f'"{k}": <1-5>' for k in DIMENSIONS)
    return _RUBRIC_SYSTEM.format(
        scale=_SCALE, dimensions=dimensions, score_keys=score_keys
    )


def score_conversation(llm: LLM, conv: Conversation, *, personas_dir: str | Path) -> Rubric:
    """Run the rubric pass on one conversation. The two profiles are read
    from ``personas_dir`` (the folder the dataset's personas live in).

    Never raises: a failed or unparseable pass returns ok=False and is
    excluded from the aggregates instead of silently scoring 0 (which
    would drag every average down and look like a quality regression).
    """
    root = Path(personas_dir)
    try:
        core_a = load_persona(
            next((root / conv.persona_a).glob("*.yaml"))
        ).core
        core_b = load_persona(
            next((root / conv.persona_b).glob("*.yaml"))
        ).core
    except Exception as e:
        # The profiles are the whole point of the distinctiveness and
        # consistency checks; scoring without them would be noise.
        return Rubric(ok=False, error=f"profiles unavailable: {e}")

    user = _RUBRIC_USER.format(
        name_a=conv.name_a,
        name_b=conv.name_b,
        core_a=core_a,
        core_b=core_b,
        ended=conv.ended or "unknown",
        transcript=conv.transcript() or "(empty conversation)",
    )
    try:
        raw = complete(
            llm,
            [
                {"role": "system", "content": _format_prompt()},
                {"role": "user", "content": user},
            ],
            temperature=0.0,  # scoring is classification; be repeatable
            # JSON mode, for the same reason judge.py uses it: without it,
            # reasoning models (DeepSeek V4 Flash) often leave `content`
            # empty or lazy and dump the real answer into
            # `message.reasoning`. Here that produced rubric passes that
            # failed with "no JSON object in rubric response" — a scoring
            # pass that looked broken — or a partial/half-hearted answer.
            # Note the asymmetry with judge.py, deliberate: the judge also
            # turns reasoning off because judging is a cheap
            # classification gate run on every single turn, whereas this
            # rubric is a demanding offline critic whose whole value is the
            # deliberation. So: force the JSON into `content`, but let the
            # model think.
            as_json=True,
        )
        data = extract_json(raw)
        if data is None:
            return Rubric(ok=False, error="no JSON object in rubric response")
        scores = {}
        for key in DIMENSIONS:
            value = data.get("scores", {}).get(key)
            if isinstance(value, (int, float)) and 1 <= value <= 5:
                scores[key] = int(round(value))
        if not scores:
            return Rubric(ok=False, error="no usable scores in rubric response")
        return Rubric(
            scores=scores,
            evidence={
                k: str(v) for k, v in (data.get("evidence") or {}).items()
            },
            verdict=str(data.get("verdict", "")),
            summary=str(data.get("summary", "")),
        )
    except Exception as e:
        logger.warning(f"rubric pass failed for {conv.id}: {e}")
        return Rubric(ok=False, error=str(e))


# --- Mechanical metrics ------------------------------------------------


def _has_repeated_ngram(text: str) -> bool:
    """True if a 4-word sequence occurs twice in the same turn."""
    words = _WORD_RE.findall(text.lower())
    seen = set()
    for i in range(len(words) - _DUP_NGRAM + 1):
        gram = tuple(words[i : i + _DUP_NGRAM])
        if gram in seen:
            return True
        seen.add(gram)
    return False


def conversation_metrics(conv: Conversation) -> dict:
    """LLM-free metrics for one conversation."""
    bubbles = [b for t in conv.turns for b in t.bubbles]
    lengths = [len(b) for b in bubbles]
    per_speaker = {"a": 0, "b": 0}
    for t in conv.turns:
        per_speaker[t.who] = per_speaker.get(t.who, 0) + 1
    tag_leaks = sum(1 for b in bubbles if _TAG_LEAK_RE.search(b))
    marker_leaks = sum(1 for b in bubbles if _MARKER_LEAK_RE.search(b))
    script_artifacts = sum(1 for b in bubbles if _SCRIPT_ARTIFACT_RE.search(b))
    repeated_turns = sum(
        1 for t in conv.turns if _has_repeated_ngram(t.text)
    )
    return {
        "turns": len(conv.turns),
        "turns_a": per_speaker.get("a", 0),
        "turns_b": per_speaker.get("b", 0),
        "bubbles": len(bubbles),
        "bubbles_per_turn": round(len(bubbles) / len(conv.turns), 2)
        if conv.turns
        else 0.0,
        "mean_bubble_chars": round(statistics.mean(lengths), 1) if lengths else 0.0,
        "max_bubble_chars": max(lengths, default=0),
        # Two different things, deliberately not merged:
        # - regenerations: drafts the judge rejected and that were then
        #   rewritten. Not a defect of the conversation (the bad draft
        #   never shipped) but a drift rate for the persona prompt.
        # - judge_final_fail: turns that were SENT anyway because the
        #   regeneration budget ran out. These are real defects, and the
        #   only ones the reader should worry about.
        "regenerations": sum(t.regenerations for t in conv.turns),
        "judge_final_fail": sum(1 for t in conv.turns if not t.judge_passed),
        # Turns the judge never saw (datasets built with --no-judge).
        # Reported explicitly so a run with no judging is never mistaken
        # for a run where every turn passed.
        "unjudged_turns": sum(1 for t in conv.turns if not t.judged),
        "violations": sorted(
            {t.judge_violation for t in conv.turns if t.judge_violation}
        ),
        # These three must be 0. Anything else is a parser bug. A turn
        # with no bubbles that LEFT is not one: the PAIRED scenario lets
        # a stranger close the chat without a word.
        "tag_leaks": tag_leaks,
        "marker_leaks": marker_leaks,
        "empty_turns": sum(
            1 for t in conv.turns if not t.bubbles and not t.left
        ),
        # Decoding artifacts. Not parser bugs — the model itself produced
        # them — but they break immersion harder than most things the
        # judge catches, so they're tracked separately and surfaced.
        "script_artifacts": script_artifacts,
        "repeated_ngram_turns": repeated_turns,
        "ended": conv.ended,
    }


def _aggregate(rows: list[dict]) -> dict:
    """Mean of every numeric metric across conversations."""
    if not rows:
        return {}
    keys = [
        k
        for k, v in rows[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return {
        k: round(statistics.mean([r[k] for r in rows]), 2) for k in keys
    }


# --- Report ------------------------------------------------------------


def _bar(value: float) -> str:
    filled = int(round(value))
    return "█" * filled + "·" * (5 - filled)


def build_report(results: list[dict], dataset_dir: Path) -> str:
    """Markdown report: aggregates first, then one block per conversation."""
    metrics = [r["metrics"] for r in results]
    rubrics = [r["rubric"] for r in results if r["rubric"].ok]
    lines = [
        "# Conversation quality report",
        "",
        f"- dataset: `{dataset_dir}`",
        f"- conversations: {len(results)}"
        + (
            f" ({len(results) - len(rubrics)} unscored)"
            if len(rubrics) != len(results)
            else ""
        ),
        "",
        "## Rubric (mean across conversations)",
        "",
    ]
    if rubrics:
        lines += ["| dimension | mean | |", "|---|---|---|"]
        for key in DIMENSIONS:
            values = [r.scores[key] for r in rubrics if key in r.scores]
            if not values:
                continue
            mean = statistics.mean(values)
            lines.append(f"| {key} | {mean:.2f} | {_bar(mean)} |")
        overall = statistics.mean(
            [r.mean for r in rubrics if r.mean is not None]
        )
        lines += ["", f"**Overall: {overall:.2f} / 5**", ""]
        verdicts: dict[str, int] = {}
        for r in rubrics:
            if r.verdict:
                verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
        if verdicts:
            lines.append(
                "Verdicts: "
                + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
            )
            lines.append("")
    else:
        lines += ["_no rubric scores (disabled or all passes failed)_", ""]

    lines += ["## Mechanical metrics (mean)", "", "| metric | value |", "|---|---|"]
    for key, value in _aggregate(metrics).items():
        lines.append(f"| {key} | {value} |")

    leaks = sum(m["tag_leaks"] + m["marker_leaks"] + m["empty_turns"] for m in metrics)
    artifacts = sum(m["script_artifacts"] for m in metrics)
    repeats = sum(m["repeated_ngram_turns"] for m in metrics)
    total_turns = sum(m["turns"] for m in metrics)
    lines += [
        "",
        (
            "✅ no format leaks (no tags, markers or empty replies reached a bubble)"
            if leaks == 0
            else f"⚠️ {leaks} format leak(s) — chat/parsing.py regression, "
                 "see per-conversation rows"
        ),
        (
            "✅ no decoding artifacts"
            if artifacts == 0 and repeats == 0
            else "⚠️ model decoding artifacts: "
                 f"{artifacts} bubble(s) with foreign-script characters, "
                 f"{repeats}/{total_turns} turn(s) with a repeated phrase"
        ),
        "",
        "## Conversations",
        "",
    ]
    for r in results:
        conv, m, rub = r["conversation"], r["metrics"], r["rubric"]
        head = f"### {conv.id} — {conv.name_a} × {conv.name_b}"
        if rub.ok and rub.mean is not None:
            head += f" ({rub.mean:.1f}/5, {rub.verdict or 'no verdict'})"
        lines += [
            head,
            "",
            f"- {m['turns']} turn(s) ({m['turns_a']}/{m['turns_b']}), "
            f"{m['bubbles']} bubbles, ~{m['mean_bubble_chars']} chars each, "
            f"ended: {m['ended']}",
        ]
        if m["unjudged_turns"]:
            # Say it plainly: a conversation with no judge data must not
            # read like one that passed every check.
            lines.append(
                f"- judge: not run on {m['unjudged_turns']} of "
                f"{m['turns']} turn(s) (dataset built with --no-judge)"
            )
        if m["regenerations"] or m["judge_final_fail"]:
            detail = f"- judge: {m['regenerations']} draft(s) regenerated"
            if m["judge_final_fail"]:
                detail += (
                    f", ⚠️ {m['judge_final_fail']} sent still failing"
                )
            if m["violations"]:
                detail += f" ({', '.join(m['violations'])})"
            lines.append(detail)
        for flag in (
            "tag_leaks",
            "marker_leaks",
            "empty_turns",
            "script_artifacts",
            "repeated_ngram_turns",
        ):
            if m[flag]:
                lines.append(f"- ⚠️ {flag}: {m[flag]}")
        if rub.ok:
            if rub.summary:
                lines += ["", f"> {rub.summary}"]
            if rub.scores:
                lines += [
                    "",
                    "| dimension | score | evidence |",
                    "|---|---|---|",
                ]
                for key in DIMENSIONS:
                    if key not in rub.scores:
                        continue
                    ev = rub.evidence.get(key, "").replace("|", "\\|")
                    lines.append(f"| {key} | {rub.scores[key]} | {ev} |")
        else:
            lines.append(f"- rubric not scored: {rub.error}")
        lines.append("")
    return "\n".join(lines)


def evaluate_dataset(
    dataset_dir: str | Path,
    *,
    llm: LLM | None = None,
    personas_dir: str | Path | None = None,
    use_rubric: bool = True,
) -> dict:
    """Evaluate every conversation of a dataset; write report + JSON.

    The rubric pass (``use_rubric``) needs ``llm`` and ``personas_dir``,
    the folder the dataset's personas live in; the mechanical metrics
    need neither."""
    dataset_dir = Path(dataset_dir)
    conversations = load_dataset(dataset_dir)
    if not conversations:
        raise ValueError(f"no conversations found in {dataset_dir}")
    if use_rubric and (llm is None or personas_dir is None):
        raise ValueError("the rubric pass needs llm and personas_dir")

    results = []
    for i, conv in enumerate(conversations, 1):
        logger.info(f"[{i}/{len(conversations)}] evaluating {conv.id}")
        rubric = (
            score_conversation(llm, conv, personas_dir=personas_dir)
            if use_rubric
            else Rubric(ok=False, error="not requested")
        )
        results.append(
            {
                "conversation": conv,
                "metrics": conversation_metrics(conv),
                "rubric": rubric,
            }
        )

    report = build_report(results, dataset_dir)
    (dataset_dir / "report.md").write_text(report, encoding="utf-8")
    payload = {
        "dataset": str(dataset_dir),
        "aggregate_metrics": _aggregate([r["metrics"] for r in results]),
        "conversations": [
            {
                "id": r["conversation"].id,
                "persona_a": r["conversation"].persona_a,
                "persona_b": r["conversation"].persona_b,
                "metrics": r["metrics"],
                "rubric": {
                    "ok": r["rubric"].ok,
                    "scores": r["rubric"].scores,
                    "evidence": r["rubric"].evidence,
                    "verdict": r["rubric"].verdict,
                    "summary": r["rubric"].summary,
                    "error": r["rubric"].error,
                },
            }
            for r in results
        ],
    }
    (dataset_dir / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"report: {dataset_dir / 'report.md'} | "
        f"json: {dataset_dir / 'evaluation.json'}"
    )
    return payload
