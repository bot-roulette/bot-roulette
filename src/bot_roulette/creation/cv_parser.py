"""CV -> persona v0.1, then the shared development pipeline.

This is ONE of two ways to create a fully functional persona; the other
is random_persona.py (invented from scratch). Both write a root persona
v0.1 folder and then run the SAME development pipeline
(pipeline.develop_persona):

1. PDF -> markdown              (PDFToMarkdown: the model reads each page)
2. markdown -> persona v0.1     (PersonaFromCV -> root persona folder)
3. AVP-style interview          (interviewer.Interviewer, INTERVIEW scenario)
4. enrichment                   (enrich.enrich_persona -> new persona folder,
                                 then stranger chats rewrite HOW YOU TEXT)

``build_persona_from_cv`` runs all of it. The result is an enriched
persona folder (e.g. ``giulia_20260720_101500/``), chat-ready once voice
calibration has rewritten its HOW YOU TEXT — or at once, with the
interview's HOW YOU TEXT, when there is no other persona to calibrate
against (see enrich.calibrate_voice). There is
no scenario-generation stage: the PAIRED entry story is built per
session, at the moment the user starts chatting (see
scenarios.live_paired_scenario). You never chat with the un-enriched
v0.1 root — only enriched personas are offered.
"""

import base64
import io
import re
import threading
from pathlib import Path

import pypdfium2 as pdfium
import yaml
from loguru import logger

from ..config import CreationConfig
from ..llm.core import GENERATION_ATTEMPTS, LLM, UnusableOutput, complete
from .pipeline import develop_persona

READ_PAGE_PROMPT = (
    "This image is page {page} of {pages} of a document. Reproduce all the "
    "text on it as markdown, verbatim and complete, in the order a person "
    "reads it: with columns or a sidebar, finish one before starting the "
    "next. Preserve section headings and lists, and write ratings drawn as "
    "graphics (dots, stars, bars) as text, e.g. \"English: 4/5\". Do not "
    "summarize, comment, translate, correct or omit anything, and do not "
    "describe pictures. Write [illegible] for anything you can't read "
    "instead of guessing. Output only the page's text, or exactly "
    "[blank page] if it has none."
)

CV_EXTRACTOR_SYSTEM_PROMPT = """You will receive a CV/résumé. Transform it into the core of a persona
file: a short, readable portrait in plain prose — the way you'd describe
a person you know, not the way a CV describes them.

STRICT RULE: use only what the CV states or what simple arithmetic
derives from it (age from graduation year, time in a city, duration of
a job — mark derived values with [EST]). Invent NOTHING: no personality,
no relationships, no habits, no reasons. The person's character will
emerge later, through interviews. If the CV doesn't say it, it doesn't
exist yet.

How to write the core:
- Flowing prose, third person, concrete and dry. "Works as a lab
  technician in a pharma company in Padua since 2021", never "results-
  oriented professional with 5+ years of experience".
- Cover, when present in the CV: age, places lived, the study path
  (including switches, interruptions, unfinished degrees — state them
  plainly, without explaining them), the work history in order, side
  jobs, volunteering, hobbies and interests, languages.
- Hobbies and volunteering are facts: report them verbatim and
  unembellished. "Climbs, volunteers at a food bank" — do NOT infer
  what this says about them.
- ALWAYS include an explicit "Languages:" line: the native language
  first (inferred from country/name/CV language if not stated — mark
  it [EST] in that case), then every other language with the
  proficiency the CV claims (keep CEFR levels verbatim if present).
  If the CV lists no other languages, say so: the person speaks
  nothing else until an interview proves otherwise.
- Close with knowledge bounds: "Knows:" from education, jobs, interests.
  Then "No evidence of:" for domains absent from the CV (this is not
  ignorance, just absence — interviews will refine it).

Self-presentation: everything the person claims about themselves
("proactive", "strong communicator", a motivational summary) is NOT
fact. Quarantine those claims, verbatim, in their own section. They
show how this person wants to be seen — the interviews will test them.

Output format — output EXACTLY two blocks:

1. A YAML header on a single line, with the person's first name and
   full name extracted from the CV:
   NAME: <first name> | FULL_NAME: <full name>
   Use the name as it appears in the CV. If only a full name is present,
   split it into first name and full name yourself. If no name is found,
   use "unknown".

2. A blank line, then the portrait:

[the portrait]
- "..." (verbatim, one per line content from the self-presentation section of the CV, if any)
"""


def parse_name_header(raw: str) -> dict:
    """Parse the ``NAME: ... | FULL_NAME: ...`` header line produced by the
    CV extractor and the random generator. Returns {name, full_name, core}.
    Raises if the header is missing: the parser never invents a name
    itself. (The CV extractor's prompt does let the model answer
    "unknown" for a CV that names nobody; that answer is kept.)
    """
    # re.search (not re.match): the model sometimes prefixes the header
    # with a fence or a line of preamble; the header is valid on ANY
    # line, and failing here throws away a paid generation.
    header_match = re.search(
        r"^NAME:\s*(.+?)\s*\|\s*FULL_NAME:\s*(.+?)\s*$",
        raw,
        re.MULTILINE,
    )
    if not header_match:
        raise RuntimeError(
            "Model output is missing the 'NAME: ... | FULL_NAME: ...' "
            f"header line. Output started with: {raw[:200]!r}"
        )
    return {
        "name": header_match.group(1).strip(),
        "full_name": header_match.group(2).strip(),
        "core": raw[header_match.end():].strip(),
    }


# Both CV steps switch the model's own reasoning off (reasoning=False):
# they copy and rewrite, there is nothing to think through. Left on, a
# reasoning model spent minutes per step here and sometimes returned its
# reasoning and no answer. Where the server can't switch it off per
# request, nothing is asked (see LLM.reasoning_off).

# Pages are rendered at twice PDF's 72 dpi, which keeps a CV's small
# print legible (an A4 page becomes 1190 x 1684 pixels), but never beyond
# _MAX_PAGE_SIDE pixels: a poster-sized page would make a huge image.
_RENDER_SCALE = 2
_MAX_PAGE_SIDE = 2000
# PDFium is not thread-safe: one document at a time, whatever the thread.
_PDFIUM_LOCK = threading.Lock()
_BLANK_PAGE_RE = re.compile(r"\W*blank page\W*", re.IGNORECASE)


def _page_images(pdf_path: str | Path) -> list[str]:
    """The PDF's pages as JPEG data URLs, leaving out blank ones."""
    images = []
    with _PDFIUM_LOCK:
        # Closed by hand: pypdfium2 4.x documents aren't context managers.
        pdf = pdfium.PdfDocument(Path(pdf_path).read_bytes())
        try:
            for page in pdf:
                scale = min(_RENDER_SCALE, _MAX_PAGE_SIDE / max(*page.get_size(), 1))
                image = page.render(scale=scale).to_pil().convert("RGB")
                if all(low == high for low, high in image.getextrema()):
                    continue  # a single colour: nothing to read
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                images.append(
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buffer.getvalue()).decode("ascii")
                )
        finally:
            pdf.close()
    return images


class PDFToMarkdown:
    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, pdf_path: str | Path) -> str:
        """Step 1: PDF -> markdown. The model reads the document one page
        at a time, as an image, so a scan reads like any other PDF (the
        model must accept images)."""
        images = _page_images(pdf_path)
        pages = []
        for n, image in enumerate(images, 1):
            logger.info(f"Reading page {n}/{len(images)}")
            text = complete(
                self.llm,
                [{"role": "user", "content": [
                    {"type": "text", "text": READ_PAGE_PROMPT.format(page=n, pages=len(images))},
                    {"type": "image_url", "image_url": {"url": image}},
                ]}],
                temperature=0,
                attempts=GENERATION_ATTEMPTS,
                reasoning=False,
            )
            if not _BLANK_PAGE_RE.fullmatch(text):
                pages.append(text)
        if not pages:
            raise UnusableOutput("found no text in the CV")
        return "\n\n".join(pages)

class PersonaFromCV:
    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, cv_markdown: str) -> dict:
        """Step 2: markdown -> persona v0.1, in a plain text call.

        Returns a dict with keys ``name``, ``full_name``, and ``core``.
        """
        raw = complete(
            self.llm,
            [
                {"role": "system", "content": CV_EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": cv_markdown},
            ],
            temperature=0,
            attempts=GENERATION_ATTEMPTS,
            reasoning=False,
        )
        return parse_name_header(raw)


def write_root_persona(name: str, full_name: str, core: str, *, root: str | Path) -> Path:
    """Write a root persona v0.1 folder (core + YAML config) into the
    personas folder ``root`` (created if missing).

    Shared by both persona sources (CV extraction here, random generation
    in random_persona.py). Returns the YAML config path.
    """
    # Use the first name as the folder/file stem (lowercased). If the
    # folder already exists (same first name), add a numeric suffix
    # instead of crashing after the paid generation.
    # Names come from model output and can contain punctuation, path
    # separators, or Windows-reserved device names. Keep all generated
    # files inside a simple child directory of ``root``.
    stem = re.sub(r"[^\w-]+", "_", name.lower(), flags=re.UNICODE).strip("_")
    if not stem:
        stem = "persona"
    windows_reserved = {
        "con", "prn", "aux", "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if stem.casefold() in windows_reserved:
        stem = f"persona_{stem}"
    # Absolute, NOT relative to the process CWD: callers find personas
    # by scanning this same folder, so a folder created relative to some
    # other working directory would never be found.
    base_dir = Path(root).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    persona_dir = base_dir / stem
    n = 2
    while persona_dir.exists():
        persona_dir = base_dir / f"{stem}{n}"
        n += 1
    stem = persona_dir.name
    persona_dir.mkdir()

    core_filename = f"{stem}.md"
    core_path = persona_dir / core_filename
    core_path.write_text(core, encoding="utf-8")
    logger.info(f"Persona core written to {core_path}")

    # Write the coordinating YAML config.
    # No "scenarios" field: scenarios are never stored per persona
    # (INTERVIEW is a shared constant, PAIRED is generated per session).
    # parent: none — this is a root persona.
    config_path = persona_dir / f"{stem}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "full_name": full_name,
                "core": core_filename,
                "parent": "none",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info(f"Persona config written to {config_path}")
    return config_path


def create_root_persona(llm: LLM, pdf_path: str | Path, *, root: str | Path) -> Path:
    """Stages 1-2: CV pdf -> persona v0.1 folder in ``root``. Returns the
    YAML config path."""
    logger.info(f"--- Transcribing CV: {pdf_path} ---")
    md = PDFToMarkdown(llm)(pdf_path)

    logger.info("--- Extracting persona v0.1 from CV ---")
    result = PersonaFromCV(llm)(md)

    return write_root_persona(
        result["name"], result["full_name"], result["core"], root=root
    )


def build_persona_from_cv(
    llm: LLM,
    pdf_path: str | Path,
    *,
    root: str | Path,
    creation: CreationConfig,
    max_turns: int | None = None,
) -> Path:
    """Full pipeline: CV pdf -> interviewed, enriched, chat-ready persona
    in the personas folder ``root``. Returns the enriched YAML config."""
    config_path = create_root_persona(llm, pdf_path, root=root)
    return develop_persona(llm, config_path, creation=creation, max_turns=max_turns)
