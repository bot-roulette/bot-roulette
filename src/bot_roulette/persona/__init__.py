"""What a persona is: a folder of plain files (spec.py) and the
situations it can be put in (scenarios.py)."""

from .scenarios import generate_entry_story, live_paired_scenario, paired_scenario
from .spec import PersonaSpec, find_chat_ready_personas, find_enriched_personas, load_persona

__all__ = [
    "PersonaSpec",
    "find_chat_ready_personas",
    "find_enriched_personas",
    "generate_entry_story",
    "live_paired_scenario",
    "load_persona",
    "paired_scenario",
]
