"""A simulated person you can text with: one Persona per conversation.

A Persona wraps a persona folder (see persona/spec.py) and a scenario into a
system prompt (framework.py) and answers one message at a time, keeping
the conversation in its own history. Each answer is checked by the
consistency judge (judge.py) unless that is turned off.
"""

from pathlib import Path

from pydantic import BaseModel

from ..llm.core import LLM, complete
from ..persona.spec import load_persona
from .framework import build_system_prompt
from .judge import Judge, Verdict
from .parsing import LEAVE_RE, format_response, parse_response, split_bubbles


def _is_usable(reply: str) -> bool:
    """Something to send: visible text, or a silent [LEAVE] (the PAIRED
    scenario lets a stranger close the chat without a word). Checked
    AFTER markers are stripped: a reply that is only "[NEXT]" is empty."""
    return bool(split_bubbles(reply)) or bool(LEAVE_RE.search(reply))


class Response(BaseModel):
    state: str
    thoughts: str
    reply: str
    left: bool = False  # persona ended the chat ([LEAVE] marker)
    verdict: Verdict | None = None  # judge verdict on the SENT response
    regenerations: int = 0          # drafts rejected by the judge this turn


class Persona:
    def __init__(
        self,
        llm: LLM,
        config_path: str | Path,
        *,
        scenario: str = "PAIRED",
        scenario_text: str | None = None,
        use_judge: bool = True,
    ):
        """``scenario_text`` supplies the scenario body directly instead
        of taking it from ``spec.scenarios``. PAIRED has no stored form
        in the persona folder — it's generated per session by
        scenarios.live_paired_scenario, so that the person opens the app
        at the moment the conversation actually starts (a resumed chat
        reuses the entry story saved with it)."""
        self.llm = llm
        self.spec = load_persona(config_path)
        if scenario_text is None and scenario not in self.spec.scenarios:
            raise ValueError(
                f"Scenario '{scenario}' not defined for persona "
                f"'{self.spec.name}'. Available: "
                f"{sorted(self.spec.scenarios) or 'none'}. "
                "PAIRED is generated per session and never stored: pass "
                "scenario_text=scenarios.live_paired_scenario(...)."
            )
        if scenario_text is not None:
            # Keep spec.scenarios authoritative for what the UI shows:
            # replace the entry, don't shadow it.
            self.spec.scenarios = dict(self.spec.scenarios)
            self.spec.scenarios[scenario] = scenario_text
        system_prompt = build_system_prompt(self.spec, scenario)
        self._history = [{"role": "system", "content": system_prompt}]
        self.temperature = 1.0
        self.judge = Judge(llm, self.spec) if use_judge else None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def history(self) -> list[dict]:
        """The full conversation state (system prompt + turns).

        Exposed so callers can persist it between sessions (the web app
        does, when chat memory is on). The returned list is the live
        history — callers treat it as read-only except for snapshotting.
        """
        return self._history

    def restore_history(self, messages: list[dict]) -> None:
        """Replace this persona's turns with a previously saved history.

        The system prompt built above is KEPT (rebuilt from the persona's
        current files plus the scenario passed to __init__ — for a resumed
        chat, the entry story saved with it); only the user/assistant
        turns of the saved history are re-applied. Passing anything
        else (a system message, a non-string content) is dropped
        defensively: history must only ever contain what the persona
        itself said or was told.
        """
        kept = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
        ]
        self._history = [self._history[0], *kept]

    def set_last_turn(self, state: str, thoughts: str, reply: str) -> None:
        """Rewrite the last assistant turn in history with edited values.

        Used by spectator mode (the spectator can edit a persona's
        state/thoughts/reply before it is fed to the other persona) and
        by the 1:1 chat's "✎ edit" (the user rewrites the persona's last
        reply). The raw model response stored in history is rebuilt from
        the edited parts so the next turn sees the edited version.

        The reply arrives bubble-per-line (that's how it's shown and
        edited in the UI); history must store it in the model's OWN
        output format — bubbles separated by [NEXT] — otherwise the
        model sees a past message in a format it never produces and its
        grounding degrades over the following turns.
        """
        if not self._history or self._history[-1]["role"] != "assistant":
            return
        bubbles = [b.strip() for b in reply.split("\n") if b.strip()]
        raw = format_response(state, thoughts, " [NEXT] ".join(bubbles))
        self._history[-1] = {"role": "assistant", "content": raw}

    @property
    def can_regenerate(self) -> bool:
        """True when history ends with a reply to a user message — the
        pair regenerate() re-asks."""
        h = self._history
        return len(h) >= 3 and h[-1]["role"] == "assistant" and h[-2]["role"] == "user"

    def regenerate(self) -> Response:
        """Replace the last reply with a fresh one to the same message.

        Used by the 1:1 chat's "↻ regenerate". The last user message is
        asked again with the last reply removed, so the new draft sees
        exactly the context the old one had (judge included). History
        only changes on success: if generation fails, the old reply
        stays. (Not to be confused with MAX_REGENERATIONS, the judge's
        redrafts inside a single call.)
        """
        if not self.can_regenerate:
            raise ValueError("there is no reply to regenerate")
        previous = self._history
        self._history = previous[:-2]
        try:
            return self(previous[-2]["content"])
        except BaseException:
            self._history = previous
            raise

    MAX_ATTEMPTS = 3       # 1 call + up to 2 retries on transient failures
    MAX_REGENERATIONS = 1  # extra drafts after a judge rejection

    def __call__(self, user_input: str) -> Response:
        messages = self._history + [{"role": "user", "content": user_input}]

        # Retry transient failures (empty content, API hiccups) so a
        # single bad completion doesn't kill the turn. The failed
        # attempt leaves no trace in history. A reply with nothing to
        # show (no reply block, or only [NEXT] markers) also counts as a
        # failure: the persona never sends nothing — except a silent
        # [LEAVE], see _is_usable.
        last_error: Exception | None = None
        state = thoughts = reply = response = None
        for _ in range(self.MAX_ATTEMPTS):
            try:
                response = complete(
                    self.llm, messages, temperature=self.temperature
                )
            except Exception as e:
                last_error = e
                continue
            state, thoughts, reply = parse_response(response)
            if _is_usable(reply):
                break
            last_error = RuntimeError("Model returned no usable reply")
        else:
            raise last_error

        # Judge pass. The judge is fail-open and biased toward PASS;
        # on a rejection we regenerate at most MAX_REGENERATIONS times
        # with the critique injected as a temporary correction (never
        # committed to history). If drafts run out, the LAST draft is
        # sent anyway: the judge can never kill or stall a turn.
        verdict: Verdict | None = None
        regenerations = 0
        if self.judge is not None:
            verdict = self.judge(messages, state, thoughts, reply)
            while not verdict.passed and regenerations < self.MAX_REGENERATIONS:
                regenerations += 1
                correction = (
                    "(Your previous draft was rejected by an internal "
                    f"consistency check — {verdict.violation}: "
                    f"{verdict.critique or 'it would expose the simulation'}. "
                    "The rejected draft was:\n"
                    f"{response}\n"
                    "Write a fresh response to the same message, fixing "
                    "only this problem. Stay fully in character and keep "
                    "the same output format.)"
                )
                retry_messages = messages + [{"role": "system", "content": correction}]
                try:
                    new_response = complete(
                        self.llm, retry_messages, temperature=self.temperature
                    )
                except Exception:
                    break  # keep the previous draft
                new_state, new_thoughts, new_reply = parse_response(new_response)
                if not _is_usable(new_reply):
                    break  # keep the previous draft
                response, state, thoughts, reply = (
                    new_response, new_state, new_thoughts, new_reply
                )
                verdict = self.judge(messages, state, thoughts, reply)

        self._history = messages + [{"role": "assistant", "content": response}]

        return Response(
            state=state,
            thoughts=thoughts,
            # One bubble per line, so every consumer can split on "\n".
            reply="\n".join(split_bubbles(reply)),
            left=bool(LEAVE_RE.search(reply)),
            verdict=verdict,
            regenerations=regenerations,
        )
