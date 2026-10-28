"""The development pipeline: persona v0.1 -> chat-ready persona.

Shared by both persona sources (cv_parser.build_persona_from_cv and
random_persona.build_random_persona): interview the v0.1 root in the
INTERVIEW scenario, then enrich it into a new folder and calibrate its
voice against stranger chats (enrich.enrich_persona). resume_development
finishes a root whose run died or was stopped.
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from ..chat.engine import Persona
from ..config import CreationConfig
from ..llm.core import LLM
from ..persona.spec import find_enriched_personas, load_persona
from .enrich import enrich_persona
from .interviewer import Interviewer


def develop_persona(
    llm: LLM,
    config_path: Path,
    *,
    creation: CreationConfig,
    max_turns: int | None = None,
) -> Path:
    """Development pipeline: persona v0.1 -> interviewed, enriched,
    voice-calibrated (when another persona exists to text), chat-ready.
    Source-agnostic: works on any root persona folder, whether it came
    from a CV or was generated from scratch.

    Returns the YAML config path of the ENRICHED persona folder. The root
    v0.1 folder is kept as the lineage parent but is not meant for chatting.
    `max_turns` (interview questions) defaults to
    creation.interview_questions.

    If the run fails or is stopped, the root folder (with its interview
    transcript, once there is one) stays on disk and the log says how to
    resume it (see resume_development).
    """
    config_path = Path(config_path)
    if max_turns is None:
        max_turns = creation.interview_questions
    try:
        return _develop(llm, config_path, max_turns=max_turns, creation=creation)
    except BaseException:
        logger.warning(
            f"{config_path.parent.name} was not finished; resume it with "
            f"resume_development (command line: bot-roulette enrich "
            f"{config_path})"
        )
        raise


def _develop(
    llm: LLM, config_path: Path, *, max_turns: int, creation: CreationConfig
) -> Path:
    # Interview against the v0.1 persona. If a transcript already exists
    # in the folder (an earlier run that died after the interview), reuse
    # it instead of paying for a new one.
    existing = sorted(config_path.parent.glob("interview_*.json"))
    if existing:
        transcript_path = existing[-1]
        logger.info(f"--- Reusing existing transcript: {transcript_path.name} ---")
    else:
        logger.info(f"--- Interviewing persona ({max_turns} questions) ---")
        persona = Persona(llm, config_path=config_path, scenario="INTERVIEW")
        interview = Interviewer(llm, max_turns=max_turns)(persona)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_path = (
            config_path.parent / f"interview_{config_path.stem}_{timestamp}.json"
        )
        transcript_path.write_text(
            json.dumps(interview, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Interview transcript saved to {transcript_path}")

    # Enrichment (field notes -> enriched folder) then voice calibration,
    # both inside enrich_persona.
    enriched_dir = enrich_persona(llm, config_path, transcript_path, creation=creation)
    enriched_config = enriched_dir / config_path.name

    if load_persona(enriched_config).chat_ready:
        logger.info(f"Persona ready: {enriched_config}")
    else:
        logger.warning(
            "Persona enriched but not chat-ready: voice calibration did not "
            f"finish; retry it with bot-roulette enrich {enriched_config}"
        )
    return enriched_config


def resume_development(llm: LLM, root_config: Path, *, creation: CreationConfig) -> Path:
    """Finish a v0.1 root whose creation run died or was stopped.

    Runs develop_persona on the root: it reuses the root's saved
    interview transcript if the run got that far, then enriches and
    calibrates. Refuses a root that was already developed (a second run
    would create a duplicate person); calibrate or rebuild the enriched
    folder instead. Returns the enriched persona's YAML config path.
    """
    root_dir = Path(root_config).resolve().parent
    for cfg in find_enriched_personas(root_dir.parent):
        if load_persona(cfg).parent == root_dir.name:
            raise ValueError(
                f"{root_dir.name} was already developed into "
                f"{cfg.parent.name}: calibrate its voice or rebuild it "
                f"instead (command line: bot-roulette enrich {cfg} "
                "[--rebuild])"
            )
    return develop_persona(llm, root_config, creation=creation)
