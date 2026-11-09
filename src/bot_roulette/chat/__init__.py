"""Talking to a persona: the Persona that answers (engine.py), its system
prompt (framework.py), the parser for its raw output (parsing.py), the
consistency judge (judge.py), two personas texting each other
(conversation.py), and the AI assistant a persona can use (assistant.py)."""

from .assistant import ASSISTANT_OPENING_NUDGE, Assistant
from .conversation import DEFAULT_MAX_TURNS, Conversation, Turn, run_conversation
from .engine import Persona, Response
from .judge import Judge, Verdict
from .parsing import LEAVE_RE, format_response, parse_response, split_bubbles

__all__ = [
    "ASSISTANT_OPENING_NUDGE",
    "DEFAULT_MAX_TURNS",
    "LEAVE_RE",
    "Assistant",
    "Conversation",
    "Judge",
    "Persona",
    "Response",
    "Turn",
    "Verdict",
    "format_response",
    "parse_response",
    "run_conversation",
    "split_bubbles",
]
