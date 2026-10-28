import json
import random
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger

from ..chat.conversation import run_conversation
from ..config import CreationConfig
from ..llm.core import GENERATION_ATTEMPTS, LLM, complete
from ..persona.spec import find_enriched_personas, load_persona


# --- Stage 1: transcript analysis -------------------------------------

ANALYST_SYSTEM_PROMPT = """You are a qualitative researcher writing up field notes after an in-depth,
AVP-style life interview. You will receive the full transcript. Your job is
to extract EVERYTHING the interview reveals about the respondent, so that a
colleague could reconstruct this person in detail without reading the
transcript.

Write structured field notes with these sections:

1. BIOGRAPHICAL FACTS — concrete, checkable details: names, ages, places,
   dates, jobs, amounts of money, family members, living situation. One
   bullet per fact. Stay literal: report what was said, not what you infer.
2. LIFE HISTORY — key events and turning points in chronological order,
   with the respondent's framing of each.
3. PEOPLE — every person mentioned: name (or role), relationship to the
   respondent, and what we know about them.
4. DAILY LIFE & HABITS — routines, schedule, food, sleep, free time,
   consumption habits.
5. INNER LIFE — values, worries, hopes, insecurities, self-image,
   contradictions between what they say and how they say it. Here careful
   inference IS allowed, but mark it as interpretation.
6. VOICE & MANNER — how they talk: tone, humor, deflections, verbal tics,
   which topics made them open up or shut down, what they refused to
   answer.
7. NOTABLE ANECDOTES — specific stories told during the interview, briefly
   summarized, since these are reusable texture.
8. TENSIONS & GAPS — anything vague, avoided, or internally inconsistent,
   and topics that were never covered.

Rules:
- Base everything on the transcript. Never invent details.
- Quote short phrases from the respondent where the wording itself is
  revealing.
- Keep the respondent's original language for quotes; write the notes in
  English.
"""



class Analyst:
    def __init__(self, llm: LLM, temperature=0.3):
        self.llm = llm
        self.temperature = temperature
    
    def __call__(self, transcript: str):
        """Stage 1: turn the raw transcript into structured field notes."""
        
        transcript = Path(transcript)
        transcript = json.loads(transcript.read_text(encoding="utf-8"))

        lines = []
        for turn in transcript:
            speaker = "INTERVIEWER" if turn["role"] == "interviewer" else "RESPONDENT"
            lines.append(f"{speaker}: {turn['content']}")
        transcript = "\n\n".join(lines)

        return complete(
            self.llm,
            [
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=self.temperature,
            attempts=GENERATION_ATTEMPTS,
        )



# --- Stage 2: persona core enrichment -----------------------------------

ENRICHER_SYSTEM_PROMPT = """You are a persona engineer. You maintain the CORE DESCRIPTION of a persona
that an LLM role-plays in text conversations. The core describes the
person's identity only: biography, personality, people, memories, voice.
The surrounding simulation machinery (hard rules, output format, scenario)
lives elsewhere and is NOT your concern — you never see it and never write it.

You will receive:
1. The CURRENT PERSONA CORE (currently in use).
2. FIELD NOTES extracted from an in-depth life interview with that same
   persona.

Your task: produce an ENRICHED version of the persona core that integrates
the interview findings.

The current core is often a dry v0.1 skeleton (facts, languages, knows /
doesn't). That skeleton is canon for identity, not a length target.
Enrichment must add the furniture a later chat can reach for: people and
scenes. A longer WHO YOU ARE paragraph of adjectives is not enrichment.

You are an ENRICHER, not a rewriter:
- Preserve existing facts, section headers already present, wording and
  voice wherever possible. Prefer ADDING and EXTENDING over rewording;
  only rephrase an existing line when weaving a new detail into it
  genuinely requires it.
- Every existing fact, trait and stylistic instruction must survive into
  the enriched core, unchanged in meaning.
- Everything the persona said in the interview is ALSO canon now. If a
  field note conflicts with the current core, the CORE wins; drop the
  conflicting detail.
- Do not paste the field notes. Do not keep every bullet. Pick the
  people, habits and scenes a stranger chat might bump into. Prefer
  specific over general. Do not invent names, places, or quotes the
  notes do not contain.

Required sections (create them if missing):
- === WHO YOU ARE === compact biography: facts, household, work, languages,
  knows / doesn't. Keep this a skeleton. Anecdotes do not belong here.
- === PERSONALITY === inner life only: values, contradictions, what stings,
  how they see themselves. Phrase interpretations as traits, not analysis
  ("you downplay how much the stalled degree weighs on you", not "subject
  exhibits avoidance"). Never put conversational duties there ("you
  answer honestly and substantively", "you do not refuse topics", "you
  volunteer information beyond what is asked"). Minimizing pain is a
  trait; answering an interviewer is not. Do not replace scenes with a
  list of adjectives.
- === YOUR PEOPLE === every recurring person from the notes (role is
  enough if unnamed): one concrete detail each — a line they said, a
  habit, how this person sits in the persona's life.
- === MEMORIES & STORIES === 4-8 short scenes, not a plot summary: who,
  where, what happened, and one quoted fragment when the wording itself
  matters. These are the core's furniture. If the notes have them, this
  section is required — not "only if genuinely needed".
- === HOW YOU TEXT === register-neutral and THIN. Tics + topic heat-map
  only (short quoted fragments like "ahah", "boh", "I don't know" are
  fine). WHEN each tic appears, what they hedge, what topics make them
  ramble or go monosyllabic. It is not how they behaved in the interview.
  NEVER write interview-stance as texting style: "you answer every
  question", "you do not refuse topics", "you volunteer beyond what is
  asked", "you are surprisingly open with a stranger", "this is the first
  time you've said this out loud". Keep this section short even when the
  rest of the core grows. A later stage rewrites it from stranger chats.

NEVER create a section of sample lines or example messages (e.g.
"=== EXAMPLES OF YOUR STYLE ==="), and do not add one if the current
core has it. Interview quotes as few-shots teach the wrong register.

Keep the core's voice: written in second person ("you") or compact
third-person biography like the original, concrete, specific, no
clinical language.

NEVER add rules, output-format instructions, scenario descriptions, or
meta commentary about how the model should behave as an AI. Identity
only.

Output format: output ONLY the new persona core text, starting directly
with the first === section header. No preamble, no markdown fences, no
commentary.
"""

class Enricher:
    def __init__(self, llm: LLM, temperature=0.3):
        self.llm = llm
        self.temperature = temperature
    
    def __call__(
        self,
        current_core: str,
        field_notes: str
    ):
        """Stage 2: merge the current persona core with the field notes."""
        user_content = (
            "=== CURRENT PERSONA CORE ===\n\n"
            f"{current_core}\n\n"
            "=== FIELD NOTES FROM THE INTERVIEW ===\n\n"
            f"{field_notes}"
        )
        return complete(
            self.llm,
            [
                {"role": "system", "content": ENRICHER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            attempts=GENERATION_ATTEMPTS,
        )


# --- Stage 3: stranger-chat voice calibration ---------------------------
#
# The interview produces a THIN HOW YOU TEXT stub (tics, topic heat-map).
# That stub is the wrong register for the product (paid researcher, not a
# stranger). After the enriched folder exists, pair the new persona with
# other ENRICHED people (calibrated or not — see _pick_voice_partners) in
# the PAIRED scenario and REPLACE the section from those transcripts.
# Identity sections are not touched.
#
# No partners (first persona in an empty folder) -> keep the stub, but
# mark the persona chat-ready anyway (chat_ready: true, voice_calibrated:
# false): one persona must be enough to start chatting. Once a partner
# exists, `bot-roulette enrich <yaml>` can still calibrate it.
# A failed pass with partners also keeps the stub, and the folder stays
# enriched but not chat-ready (chat_ready: false) until calibrate_voice
# succeeds (command line: `bot-roulette enrich <yaml>`).
#
# Sizes come from CreationConfig (creation.voice_* in config.yaml).

# The header line, then everything up to the next section or the end.
# Only the rest of the header LINE may follow the header: an earlier
# `\s*\n` also crossed blank lines, so an EMPTY section swallowed the
# next section's header and the rewrite deleted that whole section.
_HOW_YOU_TEXT_RE = re.compile(
    r"(^|\n)=== HOW YOU TEXT ===[^\n]*.*?(?=\n=== |\Z)",
    re.DOTALL,
)

VOICE_CALIBRATOR_SYSTEM_PROMPT = """You rewrite ONE section of a persona core: === HOW YOU TEXT ===.

The persona has already been interviewed. The current HOW YOU TEXT is a
thin stub of verbal tics taken from that interview. You will also receive
transcripts of this same person texting STRANGERS (a pairing-app chat
with people they have never met). Your job is to REPLACE the section so
it describes how they text a stranger, not how they answered a researcher.

Rules:
- Output ONLY the new === HOW YOU TEXT === section, starting with that
  header. No other sections, no preamble, no markdown fences.
- REPLACE, do not append. The interview stub is not canon for this
  section; the stranger transcripts are.
- Keep a tic from the stub only if it also appears in the stranger
  chats. Drop interview-stance: "answer every question", "do not refuse
  topics", "volunteer beyond what is asked", "surprisingly open",
  "first time I've said this out loud".
- Describe BEHAVIOR, not sample messages. Short quoted fragments
  ("ahah", "boh", "tipo") are fine; full example texts are not.
- Cover: language/dialect; tics and WHEN they fire; typical bubble
  length and punctuation; how they open with a stranger; what they
  leave unanswered; how guarded they are; what makes them ramble vs
  go short. Concrete and specific to THIS person.
- IGNORE LLM-LLM artifacts: a tic used in almost every turn (especially
  if the partner uses it too), matching bubble counts, messages that
  grow steadily longer, both sides asking a question every turn.
  Those are accommodation, not character.
- Do not invent biography. Do not add rules about being an AI.
- Second person ("you"), same voice as the rest of the core.
- One short paragraph, or two. Not a list of duties.
"""


class VoiceCalibrator:
    def __init__(self, llm: LLM, temperature=0.3):
        self.llm = llm
        self.temperature = temperature

    def __call__(self, current_section: str, transcripts: str) -> str:
        user_content = (
            "=== CURRENT HOW YOU TEXT ===\n\n"
            f"{current_section.strip() or '(missing)'}\n\n"
            "=== STRANGER CHAT TRANSCRIPTS ===\n\n"
            f"{transcripts}"
        )
        return complete(
            self.llm,
            [
                {"role": "system", "content": VOICE_CALIBRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            attempts=GENERATION_ATTEMPTS,
        )


def _extract_how_you_text(core: str) -> str:
    match = _HOW_YOU_TEXT_RE.search(core)
    return match.group(0).strip() if match else ""


def _normalize_how_you_text_section(new_section: str) -> str:
    """Keep only the HOW YOU TEXT block from a calibrator response."""
    new_section = new_section.strip()
    new_section = re.sub(r"^```[a-zA-Z]*\s*", "", new_section)
    new_section = re.sub(r"\s*```$", "", new_section).strip()
    idx = new_section.find("=== HOW YOU TEXT ===")
    if idx >= 0:
        new_section = new_section[idx:]
    elif not new_section.startswith("==="):
        new_section = "=== HOW YOU TEXT ===\n\n" + new_section
    extra = re.search(r"\n=== (?!HOW YOU TEXT ===)", new_section)
    if extra:
        new_section = new_section[: extra.start()].strip()
    return new_section.strip() + "\n"


def _how_you_text_body(section: str) -> str:
    return re.sub(
        r"^=== HOW YOU TEXT ===\s*", "", section.strip(), flags=re.IGNORECASE
    ).strip()


def _replace_how_you_text(core: str, new_section: str) -> str:
    """Swap HOW YOU TEXT for ``new_section``. Appends if the header is missing."""
    new_section = _normalize_how_you_text_section(new_section)
    if _HOW_YOU_TEXT_RE.search(core):
        # Function repl so LLM backslashes are never backreferences.
        return (
            _HOW_YOU_TEXT_RE.sub(lambda _: "\n" + new_section, core, count=1).strip()
            + "\n"
        )
    return core.rstrip() + "\n\n" + new_section


def _pick_voice_partners(new_config: Path, max_partners: int) -> list[Path]:
    """Up to ``max_partners`` other enriched personas from the same
    personas folder, excluding ourselves."""
    new_dir = new_config.parent.resolve()
    spec = load_persona(new_config)
    others = []
    for cfg in find_enriched_personas(new_dir.parent):
        if cfg.parent.resolve() == new_dir:
            continue
        other = load_persona(cfg)
        # A rebuild writes a sibling folder of the same person; don't
        # calibrate against another enrichment of the same root.
        if spec.parent != "none" and other.parent == spec.parent:
            continue
        others.append(cfg)
    if len(others) <= max_partners:
        return others
    # Don't always pair against the same two (glob order is alphabetical).
    return random.sample(others, max_partners)


def _speaker_transcript(conv, config_path: Path) -> str:
    """Plain transcript, this persona's lines tagged YOU (folder id, not display name)."""
    folder = config_path.parent.name
    my_who = "a" if conv.persona_a == folder else "b"
    lines = []
    for turn in conv.turns:
        tag = "YOU" if turn.who == my_who else turn.name
        text = turn.text.strip()
        if text:
            lines.append(f"{tag}: {text}")
    return "\n".join(lines)


def calibrate_voice(llm: LLM, config_path: Path, *, creation: CreationConfig) -> bool:
    """Run stranger chats and replace HOW YOU TEXT in place.

    Partners are other enriched personas in the same personas folder;
    their number and the chats' length come from ``creation``
    (voice_partners, voice_turns, voice_min_turns).

    Returns True if the core was rewritten (the persona is then
    voice-calibrated and chat-ready). False means the stub was kept: with
    no partners the persona is marked chat-ready anyway, since there is
    no one to calibrate against yet; chats too short or an empty rewrite
    leave it as it was.
    """
    spec = load_persona(config_path)
    if not spec.parent or spec.parent == "none":
        logger.warning(
            "--- Voice calibration skipped: this is a v0.1 root, not an enriched persona ---"
        )
        return False

    partners = _pick_voice_partners(config_path, creation.voice_partners)
    if not partners:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        cfg["chat_ready"] = True
        config_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info(
            "--- Voice calibration skipped: no other enriched personas yet ---"
        )
        logger.info(
            f"{spec.name} can chat with the HOW YOU TEXT it has now. Once "
            "another persona exists, calibrate the voice with: "
            f"bot-roulette enrich {config_path}"
        )
        return False

    logger.info(
        f"--- Voice calibration: {spec.name} x "
        f"{len(partners)} partner(s), up to {creation.voice_turns} turns ---"
    )

    usable: list[str] = []
    calib_dir = config_path.parent / "voice_calibration"
    calib_dir.mkdir(exist_ok=True)

    for i, partner in enumerate(partners):
        # Alternate who opens so we see both a first message and a reply
        # to someone else's opener.
        if i % 2 == 0:
            cfg_a, cfg_b = config_path, partner
        else:
            cfg_a, cfg_b = partner, config_path
        when = datetime.now() - timedelta(hours=i + 1)
        conv_id = f"voice_{i + 1:02d}"
        logger.info(
            f"--- [{i + 1}/{len(partners)}] pairing with {partner.parent.name} ---"
        )
        conv = run_conversation(
            llm,
            cfg_a,
            cfg_b,
            conv_id=conv_id,
            max_turns=creation.voice_turns,
            when=when,
            # Style sample, not a shipped chat: the judge would roughly
            # double the cost of an already-long generation job.
            use_judge=False,
        )
        (calib_dir / conv.file_name()).write_text(
            json.dumps(asdict(conv), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if len(conv.turns) < creation.voice_min_turns:
            logger.warning(
                f"[{conv_id}] only {len(conv.turns)} turns ({conv.ended}); skipping"
            )
            continue
        usable.append(
            f"--- vs {partner.parent.name} ({conv.ended}, {len(conv.turns)} turns) ---\n"
            + _speaker_transcript(conv, config_path)
        )

    if not usable:
        logger.warning("Voice calibration: no usable stranger chats; keeping stub")
        return False

    new_section = VoiceCalibrator(llm)(
        _extract_how_you_text(spec.core),
        "\n\n".join(usable),
    )
    new_section = _normalize_how_you_text_section(new_section)
    if len(_how_you_text_body(new_section)) < 40:
        logger.warning(
            "Voice calibration: calibrator output looked empty; keeping stub"
        )
        return False

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    core_rel = cfg.get("core", f"{config_path.stem}.md")
    core_path = config_path.parent / core_rel
    updated = _replace_how_you_text(core_path.read_text(encoding="utf-8"), new_section)
    core_path.write_text(updated, encoding="utf-8")
    cfg["voice_calibrated"] = True
    cfg["chat_ready"] = True
    config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info(f"HOW YOU TEXT rewritten from {len(usable)} stranger chat(s)")
    return True


def enrich_persona(
    llm: LLM,
    config_path: Path,
    transcript_path: Path,
    *,
    creation: CreationConfig,
) -> Path:
    """Full pipeline: transcript -> field notes -> enriched folder -> voice.

    Reads the persona from ``config_path`` (a YAML config), enriches its
    core with the field notes extracted from the transcript, and creates a
    NEW folder ``<source>_<timestamp>/`` containing copies of all the
    source persona's files plus the enriched core and field notes. The new
    YAML config has ``parent`` set to the source persona's folder name.

    Then, if other enriched personas exist, pairs this one with them in
    the stranger scenario and replaces ``=== HOW YOU TEXT ===`` from those
    chats. With no partners, the interview stub stays and the persona is
    chat-ready as it is; a failed pass keeps the stub too, but leaves the
    persona not chat-ready.

    Returns the path to the new folder.
    """
    spec = load_persona(config_path)
    source_dir = spec.config_path.parent
    source_name = source_dir.name

    # Stage 1: analyze transcript -> field notes
    logger.info(f"--- Analyzing transcript: {transcript_path.name} ---")
    field_notes = Analyst(llm)(str(transcript_path))

    # Stage 2: enrich the core
    logger.info(f"--- Enriching persona core: {source_name} ---")
    enriched_core = Enricher(llm)(spec.core, field_notes)

    # Create the new persona folder: <source>_<timestamp>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = source_dir.parent / f"{source_name}_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=False)

    # Copy the source folder's files (core + config; scenarios are no
    # longer stored on disk), but not transcripts or field notes: those
    # belong to the folder they were produced in, and would otherwise
    # accumulate down the lineage.
    for f in source_dir.iterdir():
        if f.is_file() and not f.name.startswith(("interview_", "field_notes_")):
            shutil.copy2(f, new_dir / f.name)

    # Overwrite the core with the enriched version.
    # The core filename is whatever the YAML's "core:" field says.
    cfg = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
    core_filename = cfg.get("core", f"{source_name}.md")
    (new_dir / core_filename).write_text(enriched_core, encoding="utf-8")

    # Save field notes in the new folder
    notes_path = new_dir / f"field_notes_{timestamp}.md"
    notes_path.write_text(field_notes, encoding="utf-8")

    # Update the YAML config: set parent to the source folder name.
    # Voice calibration (Stage 3, below) sets voice_calibrated and
    # chat_ready if it rewrites HOW YOU TEXT, chat_ready alone if there
    # is no one to calibrate against.
    cfg["parent"] = source_name
    cfg["voice_calibrated"] = False
    cfg["chat_ready"] = False
    new_config_path = new_dir / spec.config_path.name
    new_config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    logger.info(f"Enriched persona saved to {new_dir}")
    _warn_if_thin(enriched_core)

    # Stage 3: stranger chats -> replace HOW YOU TEXT. The interview
    # stub stays if there are no partners yet, or if this pass fails.
    try:
        calibrate_voice(llm, new_config_path, creation=creation)
    except Exception:
        logger.exception(
            "Voice calibration failed; keeping the interview HOW YOU TEXT stub"
        )
    return new_dir


_REQUIRED_FURNITURE = ("=== YOUR PEOPLE ===", "=== MEMORIES & STORIES ===")


def _warn_if_thin(core: str) -> None:
    missing = [h for h in _REQUIRED_FURNITURE if h not in core]
    if missing:
        logger.warning(
            f"Enriched core missing {missing}; people/scenes may have been dropped"
        )


def _latest_field_notes(folder: Path) -> Path | None:
    notes = sorted(folder.glob("field_notes_*.md"))
    return notes[-1] if notes else None


def rebuild_from_notes(llm: LLM, config_path: Path, *, creation: CreationConfig) -> Path:
    """Re-run Stage 2 from saved field notes. Writes a new enriched folder.

    Uses the v0.1 parent core as the skeleton (not the already-enriched
    file, which may have compressed people and scenes away). Voice
    calibration runs after, same as ``enrich_persona``.
    """
    spec = load_persona(config_path)
    folder = spec.config_path.parent
    if spec.parent == "none":
        raise ValueError(
            f"{config_path} is a v0.1 root; a rebuild needs an enriched "
            "folder that already has field_notes_*.md"
        )
    notes_path = _latest_field_notes(folder)
    if notes_path is None:
        raise FileNotFoundError(f"no field_notes_*.md in {folder}")

    parent_dir = folder.parent / spec.parent
    parent_config = parent_dir / spec.config_path.name
    if not parent_config.exists():
        raise FileNotFoundError(f"parent persona not found: {parent_config}")

    parent_spec = load_persona(parent_config)
    field_notes = notes_path.read_text(encoding="utf-8")
    logger.info(f"--- Re-enriching from {notes_path.name} (parent {spec.parent}) ---")
    enriched_core = Enricher(llm)(parent_spec.core, field_notes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = folder.parent / f"{spec.parent}_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=False)

    for f in parent_dir.iterdir():
        if f.is_file() and not f.name.startswith(("interview_", "field_notes_")):
            shutil.copy2(f, new_dir / f.name)

    cfg = yaml.safe_load(parent_config.read_text(encoding="utf-8"))
    core_filename = cfg.get("core", f"{spec.parent}.md")
    (new_dir / core_filename).write_text(enriched_core, encoding="utf-8")
    shutil.copy2(notes_path, new_dir / notes_path.name)

    cfg["parent"] = spec.parent
    cfg["voice_calibrated"] = False
    cfg["chat_ready"] = False
    new_config_path = new_dir / parent_config.name
    new_config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info(f"Rebuilt persona saved to {new_dir}")
    _warn_if_thin(enriched_core)

    try:
        calibrate_voice(llm, new_config_path, creation=creation)
    except Exception:
        logger.exception(
            "Voice calibration failed; keeping the interview HOW YOU TEXT stub"
        )
    return new_dir
