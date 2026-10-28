"""Random persona generation: invent a person from scratch.

This is the SECOND of two ways to create a fully functional persona
(the first is cv_parser.py, from a real CV). A generator LLM invents a
plausible, ordinary person; the result is written as a root persona
v0.1 folder and then developed by the SAME pipeline as the CV path
(pipeline.develop_persona): interview -> enrichment -> voice calibration.

Who the person is (a PersonaIdentity: first name, surname, sex, country
and, optionally, a language they must be able to chat in) is chosen by
whoever creates them. Their background is a handful of seed traits (age
band, education, area of work...) sampled with real randomness in
Python from the lists of a RandomPersonasConfig (random_personas in
config.yaml), unless the caller fixes them, so the model doesn't invent
the same life every time. A fixed trait is a value from those lists or
the caller's own words (an exact age, a named town, a precise job). The
generator builds a coherent person around both. The same lists, choices
and seed give the same seed traits.
"""

import random
import re
import unicodedata
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from ..config import CreationConfig, RandomPersonasConfig
from ..llm.core import GENERATION_ATTEMPTS, LLM, complete
from .cv_parser import parse_name_header, write_root_persona
from .pipeline import develop_persona


# --- Identity ------------------------------------------------------------
# Chosen, never drawn. Whatever is chosen is kept, even a name unusual
# for the country or the sex: the generator makes it plausible.

_IDENTITY_MAX_CHARS = {"first_name": 50, "surname": 50, "country": 60, "chat_language": 40}


class PersonaIdentity(BaseModel):
    """Who a new random persona is. ``chat_language``, if given, is a
    language they must be able to text in; otherwise they speak whatever
    is realistic for their life."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first_name: str
    surname: str
    sex: Literal["female", "male"]
    country: str
    chat_language: str | None = None

    @field_validator(*_IDENTITY_MAX_CHARS, mode="before")
    @classmethod
    def _one_line_of_text(cls, value, info: ValidationInfo):
        # Each value becomes one line of the generator's prompt.
        if not isinstance(value, str):
            return value
        value = " ".join(value.split())
        if any(unicodedata.category(c) == "Cc" for c in value):
            raise ValueError("must be plain text")
        limit = _IDENTITY_MAX_CHARS[info.field_name]
        if len(value) > limit:
            raise ValueError(f"must be at most {limit} characters")
        if not value:
            if info.field_name == "chat_language":
                return None
            raise ValueError("can't be empty")
        return value


# --- Seed traits ---------------------------------------------------------
# Sampled in Python (true randomness), then handed to the generator LLM.
# Deliberately coarse: they constrain the sketch without writing it. The
# lists come from config.yaml (random_personas). A trait the caller fixes
# can also be in their own words, as specific as they like ("Bergamo",
# "night-shift nurse"): one line of plain text, like the identity.

_NO_STABILITY = ("unemployed or between jobs", "retired")
_PRECARIOUS = "precarious (short contracts, gigs, seasonal)"

_TRAIT_OPTIONS = {
    "age_band": "age_bands",
    "education": "education",
    "work_area": "work_areas",
    "settlement": "settlements",
    "living_situation": "living_situations",
    "mobility": "mobility",
    "employment_stability": "employment_stability",
    "life_event": "life_events",
}

_CUSTOM_TRAIT_MAX_CHARS = 150
# An age in the caller's own words is still an adult's, exact or a range.
_CUSTOM_AGE_RE = re.compile(r"(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?")
_MIN_AGE, _MAX_AGE = 18, 100


def _band_start(band: str) -> int:
    return int(band.split("-")[0])


def _tidy_fixed_trait(trait: str, value: object, options: list[str]) -> str:
    """One fixed trait: a value of its list, or the caller's own words on
    one line ("37" or "40-45" for an age)."""
    if not isinstance(value, str):
        raise ValueError(f"invalid {trait}: must be text")
    value = " ".join(value.split())
    if value in options:
        return value
    if not value:
        raise ValueError(f"{trait} can't be empty")
    if any(unicodedata.category(c) == "Cc" for c in value):
        raise ValueError(f"{trait} must be plain text")
    if len(value) > _CUSTOM_TRAIT_MAX_CHARS:
        raise ValueError(f"{trait} must be at most {_CUSTOM_TRAIT_MAX_CHARS} characters")
    if trait == "age_band":
        m = _CUSTOM_AGE_RE.fullmatch(value)
        ages = [int(age) for age in m.groups() if age] if m else []
        if not ages or not _MIN_AGE <= ages[0] <= ages[-1] <= _MAX_AGE:
            raise ValueError(
                f"invalid age_band: {value!r} (an age like 37 or a range like "
                f"40-45, from {_MIN_AGE} to {_MAX_AGE})"
            )
        value = "-".join(map(str, ages))
    return value


def validate_fixed_seed_traits(
    fixed_traits: dict[str, str], lists: RandomPersonasConfig
) -> dict[str, str]:
    """Check the background traits fixed by the caller, each a value of
    its configured list or the caller's own words, and return them tidied
    (whitespace collapsed, an age range as "40-45"). Rejects unknown
    traits and fixed traits that contradict each other."""
    unknown = sorted(set(fixed_traits) - set(_TRAIT_OPTIONS))
    if unknown:
        raise ValueError(f"unknown fixed traits: {', '.join(unknown)}")

    fixed = {
        trait: _tidy_fixed_trait(trait, value, getattr(lists, _TRAIT_OPTIONS[trait]))
        for trait, value in fixed_traits.items()
    }

    # The guards key on the listed areas of work: an area in the caller's
    # own words is left for the generator to fit in.
    age_band = fixed.get("age_band")
    work_area = fixed.get("work_area")
    if work_area == "retired":
        bands = [age_band] if age_band else lists.age_bands
        if all(_band_start(band) < 55 for band in bands):
            raise ValueError("retired requires an age band starting at 55 or later")

    stability = fixed.get("employment_stability")
    if stability is not None:
        if work_area in _NO_STABILITY:
            raise ValueError(f"{work_area} cannot have fixed employment stability")
        if work_area == "student with side jobs" and stability != _PRECARIOUS:
            raise ValueError("a student with side jobs must have precarious employment")
    return fixed


def sample_seed_traits(
    rng: random.Random,
    lists: RandomPersonasConfig,
    identity: PersonaIdentity,
    fixed_traits: dict[str, str] | None = None,
) -> dict[str, str]:
    """The generator's seed traits: ``identity`` as given, plus a
    background drawn from ``lists`` except where ``fixed_traits`` fixes
    it (with a listed value or in the caller's own words). The same
    lists, choices and seed give the same traits."""
    fixed = validate_fixed_seed_traits(fixed_traits or {}, lists)

    # The age band first: it decides whether retirement is possible.
    age_band = fixed.get("age_band")
    if age_band is None:
        bands = lists.age_bands
        if fixed.get("work_area") == "retired":
            bands = [band for band in bands if _band_start(band) >= 55]
        age_band = rng.choice(bands)

    work_area = fixed.get("work_area")
    if work_area is None:
        # No retirement before 55, and a fixed employment stability needs
        # someone whose work it can describe.
        stability = fixed.get("employment_stability")
        areas = [
            area for area in lists.work_areas
            if (area != "retired" or _band_start(age_band) >= 55)
            and (stability is None or area not in _NO_STABILITY)
            and (area != "student with side jobs" or stability in (None, _PRECARIOUS))
        ]
        if not areas:
            raise ValueError("no area of work in config.yaml fits the fixed traits")
        work_area = rng.choice(areas)

    traits = {
        "sex": identity.sex,
        "age_band": age_band,
        "country": identity.country,
        "first_name": identity.first_name,
        "surname": identity.surname,
        "education": fixed.get("education") or rng.choice(lists.education),
        "work_area": work_area,
        "settlement": fixed.get("settlement") or rng.choice(lists.settlements),
        "living_situation": (
            fixed.get("living_situation") or rng.choice(lists.living_situations)
        ),
        "mobility": fixed.get("mobility") or rng.choice(lists.mobility),
        "employment_stability": (
            fixed.get("employment_stability") or rng.choice(lists.employment_stability)
        ),
        "life_event": fixed.get("life_event") or rng.choice(lists.life_events),
    }
    # Coherence guards: employment stability makes no sense for people
    # outside (stable) employment — drop the axis rather than hand the
    # generator a contradiction ("unemployed with a permanent position").
    if work_area in _NO_STABILITY:
        del traits["employment_stability"]
    elif work_area == "student with side jobs":
        traits["employment_stability"] = _PRECARIOUS
    if identity.chat_language:
        traits["chat_language"] = identity.chat_language
    return traits


# --- Generator ------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = """You invent ORDINARY people. You will receive a handful of seed traits
(sex, age band, country, first name, surname, education, area of work,
settlement type, living situation, mobility history, employment
stability, one life event, sometimes a chat language). Build one
plausible, unremarkable person around them — the kind of person a census
would find, not the kind a novelist would invent.

You produce the v0.1 core of the persona: a short factual sketch. The
person's character, opinions and inner life do NOT belong here — they
will emerge later through a long interview. Your job is only the skeleton
of a life.

Rules:
- Respect every seed trait, but make them concrete: an age band becomes
  an exact age, a country plus a settlement type becomes a specific
  NAMED town or city of that kind (not always the capital or the famous
  metropolis — pick unglamorous, real places), an area of work becomes
  a specific job with a specific employer type, the mobility history
  becomes actual named places in the right order, the living situation
  becomes the household composition, the employment stability shapes
  the work history, and the life event is woven in as one dry factual
  line (it is a fact of the record, not a story — no drama, no lesson
  learned).
- A seed trait may be written in someone's own words instead of being a
  category: an exact age, a named town, a precise job, a particular
  event. Keep every detail it gives exactly as written (the same place,
  the same job, the same event) and make concrete only what it leaves
  open. If it is unusual, keep it anyway and keep everything else
  ordinary around it. If two seed traits clash, keep the more specific
  one and adapt the other as little as possible, but never the age: the
  person is an adult of the given age.
- Aggressively ordinary: mixed results, unfinished things, small
  compromises. No prodigies, no tragic backstories, no exotic
  combinations. At most ONE mildly distinctive detail.
- The first name, surname, sex and country are FIXED, and were chosen
  by a person, not to match each other. Use the names exactly in the
  NAME / FULL_NAME header and in the sketch: do not invent a different
  name, do not translate, adapt or shorten it, do not add a middle
  name. If the name is unusual for the country or the sex, keep
  everything as given and make it plausible with one dry fact (a
  parent from abroad, a grandparent's name, a family custom).
- Stay literal and dry, like a case file written in plain prose: places
  lived, education path (including switches and interruptions, stated
  plainly), work history in order, living situation, family composition
  (counts and roles only — no relationship dynamics), 2-3 hobbies or
  habits reported flatly.
- Do NOT include: personality traits, opinions, fears, dreams, tastes
  explained by reasons, relationship dynamics, anecdotes. If it answers
  "what is this person like?", it does not belong here.
- Write the sketch in English regardless of the country. ALWAYS
  include an explicit "Languages:" line: the native language first
  (from the country/region), then any other languages with a realistic
  proficiency (e.g. "English, B1, rusty school English"). Be
  statistically honest for the country, age and education — many
  ordinary people speak little or no English, and that is fine.
- If a chat language is given, the person MUST be able to text in it
  at least well enough for a casual conversation (B1 or better). If
  it isn't their native language, give one ordinary reason they know
  it (school, work, a partner, a parent, years abroad) and list it on
  the Languages line with its level.
- Close with knowledge bounds: "Knows:" (from education, jobs, hobbies)
  and "No evidence of:" (domains absent from this life so far).

Output format — output EXACTLY two blocks:

1. A YAML header on a single line:
   NAME: <first name> | FULL_NAME: <full name>

2. A blank line, then the sketch in flowing prose, third person.
"""


def generate_random_persona(
    llm: LLM,
    identity: PersonaIdentity,
    lists: RandomPersonasConfig,
    *,
    seed: int | None = None,
    fixed_traits: dict[str, str] | None = None,
) -> dict:
    """Invent a person around ``identity``. Returns ``{name, full_name,
    core, traits}``."""
    rng = random.Random(seed)
    traits = sample_seed_traits(rng, lists, identity, fixed_traits)
    traits_text = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in traits.items())

    logger.info("--- Seed traits ---")
    logger.info(traits_text)

    raw = complete(
        llm,
        [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Seed traits:\n{traits_text}"},
        ],
        temperature=1.0,
        attempts=GENERATION_ATTEMPTS,
    )
    result = parse_name_header(raw)
    # The name is chosen, not the model's. The header is still parsed
    # (the sketch must start after it), but the identity comes from
    # Python, so a model that "corrects" an unusual name cannot leak its
    # own: not as the full name, and not as a bare surname ("the Rossi
    # household") either.
    chosen_name = identity.first_name
    chosen_full = f"{identity.first_name} {identity.surname}"
    model_name, model_full = result["name"], result["full_name"]
    model_surname = (
        model_full[len(model_name):].strip()
        if model_full.startswith(model_name + " ")
        else ""
    )
    core = result["core"]
    core = _swap_name(core, model_full, chosen_full)
    core = _swap_name(core, model_surname, identity.surname)
    core = _swap_name(core, model_name, chosen_name)
    result["name"] = chosen_name
    result["full_name"] = chosen_full
    result["core"] = core
    result["traits"] = traits
    return result


def _swap_name(text: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` as a whole word only: a plain
    str.replace also rewrites names inside longer words."""
    if not old or old == new:
        return text
    return re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", lambda _: new, text)


def build_random_persona(
    llm: LLM,
    *,
    identity: PersonaIdentity,
    root: str | Path,
    creation: CreationConfig,
    random_personas: RandomPersonasConfig,
    seed: int | None = None,
    max_turns: int | None = None,
    fixed_traits: dict[str, str] | None = None,
) -> Path:
    """Full pipeline: a person built around ``identity``, with a
    background drawn from ``random_personas`` except where
    ``fixed_traits`` fixes it -> interviewed, enriched, chat-ready
    persona in the personas folder ``root``. Returns the YAML config path
    of the enriched folder. `max_turns` defaults to
    creation.interview_questions."""
    logger.info("--- Generating random persona v0.1 ---")
    result = generate_random_persona(
        llm, identity, random_personas, seed=seed, fixed_traits=fixed_traits
    )
    config_path = write_root_persona(
        result["name"], result["full_name"], result["core"], root=root
    )

    # Record the seed traits in the YAML for provenance.
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["seed_traits"] = result["traits"]
    config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    return develop_persona(llm, config_path, creation=creation, max_turns=max_turns)
