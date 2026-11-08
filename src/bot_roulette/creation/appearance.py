"""What a persona looks like: asked before the interview, kept in the core.

The v0.1 root is asked to describe itself, in character and in its own
language: in the INTERVIEW scenario a researcher asks, so it answers. A
describer turns the answer, checked against the core, into one line of
physical facts that joins the core as "Looks: ...". Everything after it
sees the same face: the interview, the enrichment, every chat, the judge
and the portrait, which is drawn from that very line.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

from ..chat.engine import Persona
from ..llm.core import GENERATION_ATTEMPTS, LLM, complete
from ..llm.images import ImageModel, generate_image
from ..persona.spec import load_persona

APPEARANCE_QUESTION = (
    "Before we start, one thing for the study's records: we ask every "
    "participant to describe what they look like, in their own words. How "
    "would you describe yourself physically? Your face, hair, eyes, skin, "
    "build, anything people tend to notice about you."
)

TRANSLATOR_SYSTEM_PROMPT = """Translate the question you receive into {language}, as a researcher
would ask it of a study participant: the same meaning, plain and polite.
Output only the translation.
"""

DESCRIBER_SYSTEM_PROMPT = """You record what a person looks like. You will receive their CORE
DESCRIPTION (who they are) and THEIR OWN ANSWER when asked what they
look like. Write it up as one English noun phrase, the way a careful
observer would describe them to a portrait painter: it goes into their
file, and it is the subject of their photo ("A photo of a ...").

Rules:
- Start with their age and sex, e.g. "41-year-old man".
- Their answer is canon for their looks: keep every physical detail it
  gives, in plain visual terms. Self-deprecation becomes what it would
  look like ("I always look tired" -> "faint shadows under the eyes"),
  never a judgment.
- The core is canon for facts: where the answer contradicts it (a
  different age), the core wins.
- Fill in whatever the answer leaves out: height and build, skin tone,
  face shape, eyes and eyebrows, nose, hair (colour, length, texture,
  hairline), facial hair, glasses, visible marks. Choose what is most
  plausible for their age, origin and life as the core tells it. If the
  answer says nothing about their looks (a refusal, a joke), build it
  all from the core.
- Plain words, no flattery: an ordinary person, as they are.
- Physical facts only: no name, personality, mood, job or story, and no
  expression, pose, clothing, background or lighting.
- 40 to 80 words. Output only the phrase: no quotes, no preamble, no
  full stop at the end.
"""

PORTRAIT_PROMPT = """
A professional, high-definition 35x45mm biometric passport photo of a {physical_characteristics}, looking directly at the camera. The face is perfectly centered and shot from the front, covering 70-80% of the frame from the chin to the forehead. Sharp focus, clear details, high-quality print texture. An ordinary person, not a model: natural skin texture and small asymmetries, no retouching. Neutral expression with mouth closed and eyes open and clearly visible. Shot on a solid, uniform white background with even, soft studio lighting. No flash reflections on the face, no red-eye effect. The subject is alone with no other people or objects in the frame. Full color.
""".strip()

PORTRAIT_ASPECT_RATIO = "3:4"  # the ratio nearest 35x45 mm that nearly every image model offers

_LOOKS_RE = re.compile(r"^Looks:[ \t]*(.+)$", re.MULTILINE)
_LANGUAGES_RE = re.compile(r"^Languages:", re.MULTILINE)
# The first language listed, the native one: "Languages: Italian (native), English (B1)".
_NATIVE_LANGUAGE_RE = re.compile(r"\bLanguages:[ \t]*([^,;(\[\n]+)")


def describe_appearance(llm: LLM, config_path: Path) -> str:
    """The persona's looks: the core's "Looks:" line, asked for and
    written into the core first if it has none. The exchange is saved
    next to the core, as appearance_<name>_<timestamp>.json."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    core_path = config_path.parent / cfg.get("core", f"{config_path.stem}.md")
    core = core_path.read_text(encoding="utf-8")
    existing = _LOOKS_RE.search(core)
    if existing:
        logger.info("--- Reusing the looks already in the core ---")
        return existing.group(1).strip()

    persona = Persona(llm, config_path, scenario="INTERVIEW")
    logger.info(f"--- Asking {persona.name} what they look like ---")
    question = _in_their_language(llm, core)
    answer = persona(question).reply
    logger.info(answer)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exchange = [
        {"role": "interviewer", "content": question},
        {"role": "respondent", "content": answer},
    ]
    (config_path.parent / f"appearance_{config_path.stem}_{timestamp}.json").write_text(
        json.dumps(exchange, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    looks = _describe(llm, core, answer)
    core_path.write_text(_with_looks(core, looks), encoding="utf-8")
    logger.info(f"Looks: {looks}")
    return looks


def _in_their_language(llm: LLM, core: str) -> str:
    """APPEARANCE_QUESTION in the persona's native language, the first on
    its Languages line: before the interview, there is no interviewer to
    switch to it. The language is picked here, not by the model: asked
    to pick it, a model once put an English speaker's question in Arabic."""
    found = _NATIVE_LANGUAGE_RE.search(core)
    language = found.group(1).strip(" .") if found else ""
    if language.casefold() in ("", "english"):
        return APPEARANCE_QUESTION
    return complete(
        llm,
        [
            {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": APPEARANCE_QUESTION},
        ],
        temperature=0,
        reasoning=False,
        attempts=GENERATION_ATTEMPTS,
    )


def _describe(llm: LLM, core: str, answer: str) -> str:
    """The answer, checked against the core and completed, as one noun
    phrase (see DESCRIBER_SYSTEM_PROMPT)."""
    phrase = complete(
        llm,
        [
            {"role": "system", "content": DESCRIBER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"=== CORE DESCRIPTION ===\n{core}\n\n"
                    f"=== THEIR OWN ANSWER ===\n{answer}"
                ),
            },
        ],
        temperature=0.3,
        reasoning=False,
        attempts=GENERATION_ATTEMPTS,
    )
    # One line, and it goes mid-sentence: "...photo of a <phrase>, looking..."
    phrase = " ".join(phrase.split()).strip("\"'“”").rstrip(".")
    return re.sub(r"^an?\s+", "", phrase, flags=re.IGNORECASE)


def _with_looks(core: str, looks: str) -> str:
    """``core`` with its Looks line: just before the Languages line, or
    at the end if no line starts with "Languages:"."""
    line = f"Looks: {looks}"
    languages = _LANGUAGES_RE.search(core)
    if languages is None:
        return f"{core.rstrip()}\n\n{line}\n"
    return f"{core[:languages.start()]}{line}\n\n{core[languages.start():]}"


def draw_portrait(image_model: ImageModel, config_path: Path, looks: str) -> None:
    """The persona's passport photo, drawn from its looks into
    portrait.<ext> next to the core, unless it has one already.
    Fail-open: a refused or failed image is logged, and the persona is
    complete without it."""
    if load_persona(config_path).portrait is not None:
        return
    logger.info("--- Drawing the portrait ---")
    try:
        image, extension = generate_image(
            image_model,
            PORTRAIT_PROMPT.format(physical_characteristics=looks),
            aspect_ratio=PORTRAIT_ASPECT_RATIO,
        )
    except Exception as e:
        logger.warning(f"No portrait ({e}); the persona is complete without one")
        return
    path = Path(config_path).parent / f"portrait.{extension}"
    path.write_bytes(image)
    logger.info(f"Portrait saved to {path}")
