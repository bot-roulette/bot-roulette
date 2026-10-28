"""Two personas texting each other, headless.

Runs the same machinery as the web app's spectator mode — two `Persona`
instances taking turns, each one's reply fed to the other as a user
message — without a UI, and returns the whole exchange as a
``Conversation`` record. Voice calibration (creation/enrich.py) uses it
for its stranger chats, and evaluation/dataset.py for its batches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..llm.core import LLM
from ..persona.scenarios import live_paired_scenario
from ..persona.spec import load_persona
from .engine import Persona

# Long enough for a conversation to develop a shape (and to fizzle),
# short enough that a full sweep stays affordable. Deliberately LOWER
# than the web spectator limit (web.spectate_turns in config.yaml, 40
# as shipped): a dataset batch (evaluation/dataset.py) multiplies this
# by pairs x repeats, so the ceiling matters far more there.
DEFAULT_MAX_TURNS = 30

# The opener has to speak into an empty window; without a nudge the
# model waits for a message that will never come. Kept identical in
# spirit to the web app's spectator nudge — a conversation collected
# here should be the same object the web app produces.
OPENING_NUDGE = (
    "(system note: you've just been paired. the chat window is empty — "
    "the other person hasn't said anything yet. it's on you to send the "
    "first message.)"
)

# Turn-level retries, on top of the ones Persona.__call__ already does.
# Losing turn 15 of 30 to one transient provider error wastes the whole
# conversation, which is the single most expensive unit of a batch run.
TURN_ATTEMPTS = 2


# --- Records -----------------------------------------------------------
# Plain dataclasses rather than pydantic models: these are inert
# serialization shapes, and asdict() gives the JSON for free.


@dataclass
class Turn:
    index: int
    who: str            # "a" or "b"
    name: str           # speaker's display name
    state: str
    thoughts: str
    bubbles: list[str]  # the reply, split the way the UI shows it
    left: bool = False  # this turn tapped the leave button
    # Whether the consistency judge actually ran on this turn. Without
    # this flag, `--no-judge` runs recorded judge_passed=True for every
    # turn and were indistinguishable from "the judge found nothing",
    # so the report showed a perfect score that meant nothing.
    judged: bool = True
    judge_passed: bool = True
    judge_violation: str = ""
    regenerations: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.bubbles)


@dataclass
class Conversation:
    id: str
    persona_a: str          # folder id
    persona_b: str
    name_a: str
    name_b: str
    moment: str             # ISO timestamp the entry stories were built on
    scenario_a: str
    scenario_b: str
    turns: list[Turn] = field(default_factory=list)
    ended: str = ""         # "left" | "turn_limit" | "error"
    ended_detail: str = ""

    def transcript(self) -> str:
        """Human-readable transcript: only what the two personas saw."""
        return "\n".join(f"{t.name}: {t.text}" for t in self.turns)

    def file_name(self) -> str:
        """The JSON file this conversation is saved as: its id and both
        speakers' names. Names come from model output (a CV), so they
        are reduced to characters that are safe in a file name."""
        a, b = (re.sub(r"[^\w-]+", "_", n).strip("_") or "persona"
                for n in (self.name_a, self.name_b))
        return f"{self.id}_{a}__{b}.json"


# --- Generation --------------------------------------------------------


def run_conversation(
    llm: LLM,
    config_a: Path,
    config_b: Path,
    *,
    conv_id: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    when: datetime | None = None,
    use_judge: bool = True,
) -> Conversation:
    """One headless conversation between two personas.

    Both entry stories are generated from the SAME `when`: independently
    generated ones disagree about the time of day (one at her desk at
    16:40, the other on the sofa at 21:40) and the personas then argue
    about it on air, which poisons the sample.

    Never raises for a failure DURING the conversation: a failure
    mid-conversation is recorded in `ended` and whatever turns were
    collected are kept. A batch run must not die on turn 19 of 30 and
    lose everything before it. (Loading the two persona configs happens
    before the guard, so a broken YAML still raises — that is a setup
    error, not a conversation failure.)
    """
    when = when or datetime.now()
    spec_a, spec_b = load_persona(config_a), load_persona(config_b)
    conv = Conversation(
        id=conv_id,
        persona_a=config_a.parent.name,
        persona_b=config_b.parent.name,
        name_a=spec_a.name,
        name_b=spec_b.name,
        moment=when.isoformat(timespec="minutes"),
        scenario_a="",
        scenario_b="",
    )
    try:
        conv.scenario_a = live_paired_scenario(llm, spec_a.core, when=when)
        conv.scenario_b = live_paired_scenario(llm, spec_b.core, when=when)
        persona_a = Persona(
            llm,
            config_path=config_a,
            scenario_text=conv.scenario_a,
            use_judge=use_judge,
        )
        persona_b = Persona(
            llm,
            config_path=config_b,
            scenario_text=conv.scenario_b,
            use_judge=use_judge,
        )
    except Exception as e:
        logger.exception(f"[{conv_id}] setup failed")
        conv.ended, conv.ended_detail = "error", f"setup: {e}"
        return conv

    speaker, who = persona_a, "a"
    text = OPENING_NUDGE
    for i in range(max_turns):
        # Persona.__call__ already retries a few times, but a provider
        # hiccup can outlast its budget and lose the whole rest of the
        # conversation (the first real run died on turn 15 of 30 on a
        # single "Model returned empty content"). One more retry here,
        # at the turn level, is cheap insurance: a failed attempt leaves
        # no trace in the persona's history, so retrying is safe.
        answer = None
        last_error: Exception | None = None
        for attempt in range(TURN_ATTEMPTS):
            try:
                answer = speaker(text)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{conv_id}] turn {i} attempt {attempt + 1}/"
                    f"{TURN_ATTEMPTS} failed: {e}"
                )
        if answer is None:
            logger.error(f"[{conv_id}] turn {i} failed: {last_error}")
            conv.ended, conv.ended_detail = "error", f"turn {i}: {last_error}"
            return conv

        turn = Turn(
            index=i,
            who=who,
            name=speaker.spec.name,
            state=answer.state,
            thoughts=answer.thoughts,
            bubbles=[b for b in answer.reply.split("\n") if b.strip()],
            left=answer.left,
            judged=answer.verdict is not None,
            judge_passed=answer.verdict.passed if answer.verdict else True,
            judge_violation=(
                answer.verdict.violation if answer.verdict else ""
            ),
            regenerations=answer.regenerations,
        )
        conv.turns.append(turn)
        logger.info(f"[{conv_id}] {turn.name}: {turn.text}")

        if answer.left:
            conv.ended = "left"
            conv.ended_detail = f"{speaker.spec.name} left the chat"
            return conv

        # Hand the reply to the other persona (its bubbles arrive as one
        # user message, exactly as in the spectator path).
        text = answer.reply
        speaker, who = (persona_b, "b") if who == "a" else (persona_a, "a")

    conv.ended = "turn_limit"
    return conv
