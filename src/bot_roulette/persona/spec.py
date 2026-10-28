"""Persona loading from plain files (no registry, no JSON store).

A persona lives in its own folder (e.g. ``giulia/``) containing:

- ``<name>.yaml``      — the coordinating config (see below).
- ``<name>.md``        — the persona CORE (biography, personality, people,
  memories, texting style). Mutable: evolved by the Enricher (enrich.py).

The YAML config:

.. code-block:: yaml

    name: Giulia
    full_name: Giulia Marchetti
    core: giulia.md            # relative to the YAML's folder
    parent: none               # persona this was enriched from ("none" = root)
    voice_calibrated: false    # true only after HOW YOU TEXT was rewritten
                               # from stranger chats
    chat_ready: false          # offered for chatting (chat front ends require
                               # this): voice-calibrated, or calibration was
                               # skipped because there was no one to text

When the Enricher creates a new persona, it writes a new folder
(e.g. ``giulia_20260714_204800/``) with copies of all files plus the
enriched core, and sets ``parent`` to the source persona's folder name.
Persona folders therefore sit side by side in one folder (the apps use
``personas/``): that is the folder the finders below scan, and where
new and enriched personas are written.

The simulation machinery (hard rules, output format, prompt assembly)
lives in framework.py and is persona-agnostic. Scenarios are not
persona data: see scenarios.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .scenarios import DEFAULT_SCENARIOS


@dataclass
class PersonaSpec:
    """Everything that defines one simulated person, loaded from files."""

    name: str                  # first name, used in prompts and UI
    full_name: str             # full name, used in the prompt header
    core: str                  # identity description (mutable, from .md)
    scenarios: dict[str, str]  # scenario name -> situation block
    config_path: Path | None = None  # path to the coordinating YAML
    parent: str = "none"       # persona this one was enriched from ("none" = root)
    voice_calibrated: bool = False  # HOW YOU TEXT rewritten from stranger chats
    chat_ready: bool = False   # offered for chatting (see find_chat_ready_personas)


def load_persona(config_path: str | Path) -> PersonaSpec:
    """Load a persona from its coordinating YAML config file.

    The YAML must contain ``name``, ``full_name``, and ``core`` (a path
    to the core markdown file, relative to the YAML).
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Persona config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent

    name = cfg.get("name", config_path.stem)
    full_name = cfg.get("full_name", name)
    parent = cfg.get("parent", "none")
    voice_calibrated = bool(cfg.get("voice_calibrated", False))
    # Calibrated implies ready (folders older than chat_ready lack the key).
    chat_ready = voice_calibrated or bool(cfg.get("chat_ready", False))

    # Core: required, resolved relative to the YAML file.
    core_rel = cfg.get("core")
    if not core_rel:
        raise ValueError(
            f"Persona config {config_path} is missing required 'core' field"
        )
    core_path = (base_dir / core_rel).resolve()
    if not core_path.exists():
        raise FileNotFoundError(f"Persona core not found: {core_path}")
    core = core_path.read_text(encoding="utf-8").strip()

    # Scenarios are not persona data: INTERVIEW is a shared constant and
    # PAIRED is generated per session (see scenarios.py).
    scenarios: dict[str, str] = dict(DEFAULT_SCENARIOS)

    return PersonaSpec(
        name=name,
        full_name=full_name,
        core=core,
        scenarios=scenarios,
        config_path=config_path,
        parent=parent,
        voice_calibrated=voice_calibrated,
        chat_ready=chat_ready,
    )


def find_enriched_personas(root: str | Path) -> list[Path]:
    """All ENRICHED persona configs in the folder ``root``
    (``root/<persona>/<name>.yaml``).

    Enriched means interviewed and woven into a core (YAML ``parent`` is
    the v0.1 folder, not ``none``). That is enough to be a voice-calibration
    partner or a dataset speaker. A chat picker should only offer a subset —
    see ``find_chat_ready_personas``. Unreadable folders are skipped.
    """
    base = Path(root)
    ready = []
    for cfg in sorted(base.glob("*/*.yaml")):
        try:
            spec = load_persona(cfg)
        except Exception:
            continue
        if spec.parent and spec.parent != "none":
            ready.append(cfg)
    return ready


def find_chat_ready_personas(root: str | Path) -> list[Path]:
    """Enriched personas that can be offered for chatting (``chat_ready``).

    That is every persona whose ``HOW YOU TEXT`` was rewritten from
    stranger chats, plus any whose voice calibration was skipped because
    there was no one to text: the first persona in an empty folder keeps
    its interview ``HOW YOU TEXT`` rather than wait for a second one (see
    enrich.calibrate_voice). A persona whose calibration failed or was
    stopped while partners existed stays out until a re-run succeeds.
    Chats should use this list; voice-partner picking uses
    ``find_enriched_personas``.
    """
    return [
        cfg
        for cfg in find_enriched_personas(root)
        if load_persona(cfg).chat_ready
    ]


