"""Persona chat web server: pick a persona and chat with them.

Design:
- One WebSocket connection = one chat session. All state (the persona's
  message history) lives in memory on the connection; when the socket
  closes, the session is gone — unless CHAT_MEMORY_ALLOWED saved it
  after every turn (see chats.py). No database.
- The client first fetches GET /personas to show the picker (id, name,
  full core text for the preview). Its FIRST WebSocket message then
  names the chosen persona: {"type": "start", "persona": id}.
- Spectator mode: if the start message also has "persona_b", the two
  personas chat with EACH OTHER and the user just watches. Bubbles and
  inner monologues carry "who" ("a"/"b") and "name"; the client can
  send {"type": "stop"} at any time to end the show.
- New personas can be created from the picker:
    GET  /personas/generation-options         the background lists (config.yaml)
    POST /personas/generate                   invent a person around a given identity:
      {"identity": {"first_name": ..., "surname": ..., "sex": "female"|"male",
                    "country": ..., "chat_language": ... (optional)},
       "fixed_traits": {"age_band": "25-34", ...}}   optional; the rest is random.
      A fixed trait is a listed value or your own words ("settlement":
      "Bergamo"); an age in your own words is "37" or "40-45", 18 to 100.
    POST /personas/import-cv                  multipart PDF upload
  Both run the full development pipeline (interview -> enrichment ->
  voice calibration) in a background thread; the client polls
  GET /jobs/{job_id} until status is "done", "error" or
  "cancelled". POST /jobs/{job_id}/cancel stops a job at its
  next model call. Only chat-ready personas (voice-calibrated, or
  created when there was no other persona to calibrate against) are
  offered for chatting; each session generates its own PAIRED entry story
  (a resumed chat reuses the one saved with it).
- Where the model runs, the model and its API key (the "⚙ settings"
  panel): OpenRouter, or your own OpenAI-compatible server (LLM_BASE_URL),
  local or remote:
    GET  /settings                    where it runs, the model, hints of the keys
    GET  /settings/models[?server=]   suggestions: OpenRouter's catalog, or a server's models
    POST /settings                    {"server": ..., "model": ..., "api_key": ...}
  A POST checks each changed value with OpenRouter or your server (free
  calls), writes it to .env and applies it at once: the next model call
  in every chat and creation job uses it. An empty field keeps its
  value, and the keys themselves are never sent back. A key only goes
  to the server it was saved for. Until the model (and, on OpenRouter,
  the key) is set, the server still starts but refuses chats and persona
  creation with a pointer to the panel.
- Replies are split into bubbles ([NEXT]), one message each.
- Optional access code (ACCESS_CODE in .env). When set, every request
  needs it as the HTTP Basic password, the chat WebSocket included: the
  browser asks for it once in its own popup (the username is ignored)
  and then sends it by itself. Unset, the app is open to anyone who can
  reach it; it listens on 127.0.0.1 by default (see __main__.py).
- The start message also carries "local_time" (the client's naive local
  ISO timestamp): the persona's entry story is generated for the time of
  day the USER is living, since the server clock is UTC in production.
  It may also carry "chat_id" to resume a saved chat.
- After the start message the server sends:
    {"type": "typing"}                        setting the scene
    {"type": "matched", "name": ...}          chat started
    {"type": "core", "name": ..., "text": ..., "scenario": ...}
                                              the "who is this?" panel
  and, with chat memory on, a resumed chat's saved turns as
    {"type": "replay", "who": "me"|"them", "text": ...}
    {"type": "replay", "who": "inner", "state": ..., "thoughts": ...}
  followed by {"type": "chat", "id": ..., "resumed": ..., "turn": ...}.
- The client then sends {"type": "message", "text": ...} and receives:
    {"type": "typing"}                        persona is typing
    {"type": "inner", "state": ..., "thoughts": ...}  inner monologue
    {"type": "bubble", "text": ...}           one message bubble
    {"type": "done", "turn": ...}             reply complete
    {"type": "ended", "reason": ...}          session over
- Until it sends the next message, the client may rewrite the persona's
  last reply: {"type": "edit", "turn": ..., "state": ..., "thoughts":
  ..., "reply": ...}, where "turn" is the id from "done" (or "chat",
  after a resume) and "reply" has one bubble per line. The server
  answers {"type": "edited", "turn": ..., "state": ..., "thoughts": ...,
  "bubbles": [...]} with the turn as stored, or {"type":
  "edit_rejected", "turn": ..., "reason": ...} (stale turn, empty reply).
- Or replace it with a fresh reply to the same message: {"type":
  "regenerate", "turn": ...}. The server answers {"type": "typing"},
  {"type": "regenerated", "turn": ...} (drop that reply), then the new
  one as usual ("inner", "bubble"s, "done" with a NEW turn id) — or
  {"type": "regenerate_rejected", "turn": ..., "reason": ...} (stale
  turn, failed generation), keeping the old reply. Regenerating is not
  capped and doesn't count toward the session's message limit.
  Turn ids change with every new reply, so an edit or regenerate meant
  for a replaced reply is always stale.
- In spectator mode, after each turn the server sends {"type":
  "editable", "who": ..., "name": ..., "state": ..., "thoughts": ...,
  "reply": ...} and waits for {"type": "continue"} or {"type": "edit",
  "who": ..., "state": ..., "thoughts": ..., "reply": ...}.

Run locally, from the home folder (see home.py):

    bot-roulette-web                    # or: uvicorn bot_roulette_web.server:app --reload

Then open http://localhost:8000
"""

import asyncio
import base64
import hmac
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from bot_roulette.chat import LEAVE_RE, Persona, parse_response, split_bubbles
from bot_roulette.config import Config, load_config
from bot_roulette.creation import (
    PersonaIdentity,
    build_persona_from_cv,
    build_random_persona,
    validate_fixed_seed_traits,
)
from bot_roulette.llm import (
    Cancelled,
    OpenRouterUnavailable,
    ServerUnavailable,
    check_api_key,
    check_model,
    check_server_model,
    is_known_model,
    is_served_model,
    model_catalog,
    server_models,
    set_cancel_event,
)
from bot_roulette.persona import (
    find_chat_ready_personas,
    find_enriched_personas,
    generate_entry_story,
    load_persona,
    paired_scenario,
)

from . import home
from .settings import (
    is_configured,
    load_env,
    make_llm,
    save_settings,
    server_api_key,
    server_url,
    settings_summary,
)


class WebConfig(BaseModel):
    """config.yaml's web: section (see its comments)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judge: bool
    messages_per_session: PositiveInt
    message_chars: PositiveInt
    spectate_turns: PositiveInt
    cv_max_mb: PositiveInt


class AppConfig(Config):
    """The library's sections (creation, random_personas) plus web."""

    web: WebConfig


class GeneratePersonaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: PersonaIdentity
    fixed_traits: dict[str, str] = Field(default_factory=dict)


# .env first: everything below reads the environment it fills in. The
# library is quiet by default; its progress logs are this app's console
# output and the creation jobs' progress log.
load_env()
logger.enable("bot_roulette")
CONFIG = load_config(home.CONFIG_PATH, schema=AppConfig)

# Tunables come from config.yaml (see its comments for what each does).

# A chat ends after this many user messages (regenerations don't count).
MAX_MESSAGES_PER_SESSION = CONFIG.web.messages_per_session

# A single chat message (or edit) longer than this is truncated before
# reaching the persona: unbounded input would balloon the prompt (and
# the bill).
MAX_MESSAGE_CHARS = CONFIG.web.message_chars

# Uploaded CVs above this size are rejected up front.
MAX_PDF_MB = CONFIG.web.cv_max_mb
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024

# Spectator mode: two personas talk to each other while the user watches.
# Each persona reply counts as one turn; the session ends when a persona
# leaves, the limit is hit, or the spectator stops watching.
SPECTATE_MAX_TURNS = CONFIG.web.spectate_turns

# The consistency judge on web replies (chats and spectator mode).
JUDGE_WEB_CHATS = CONFIG.web.judge

# Out-of-band nudge that kicks off a spectated conversation: the first
# persona needs a reason to speak into an empty chat window.
SPECTATE_OPENING_NUDGE = (
    "(system note: you've just been paired. the chat window is empty — "
    "the other person hasn't said anything yet. it's on you to send the "
    "first message.)"
)

# Persona creation (random generation / CV import) is opt-in: it burns many
# LLM calls, so it must be explicitly enabled via the environment.
GENERATION_ALLOWED = os.environ.get("GENERATION_ALLOWED", "").strip().lower() in (
    "true", "1", "yes", "on",
)

# Persona deletion is opt-in: it permanently removes a persona folder,
# so it must be explicitly enabled via the environment.
CANCELLATION_ALLOWED = os.environ.get("CANCELLATION_ALLOWED", "").strip().lower() in (
    "true", "1", "yes", "on",
)

# Chat memory is opt-in: when on, each chat is saved to disk and can be
# resumed later. Chats are strictly independent — a resumed chat only
# ever sees its own saved history (see chats.py).
CHAT_MEMORY_ALLOWED = os.environ.get("CHAT_MEMORY_ALLOWED", "").strip().lower() in (
    "true", "1", "yes", "on",
)

# Optional access code: with ACCESS_CODE set, the browser asks for it
# before showing anything (see require_access_code). Unset or empty, the
# app is open to anyone who can reach it.
ACCESS_CODE = os.environ.get("ACCESS_CODE", "")

app = FastAPI()
llm = make_llm()


def _access_ok(headers) -> bool:
    """True when no ACCESS_CODE is set, or the request carries it as the
    password of HTTP Basic auth (the username is ignored)."""
    if not ACCESS_CODE:
        return True
    scheme, _, encoded = headers.get("authorization", "").partition(" ")
    try:
        password = base64.b64decode(encoded).decode("utf-8").partition(":")[2]
    except ValueError:  # not base64, or not UTF-8
        return False
    # Constant time, and as bytes: compare_digest raises on non-ASCII str.
    return scheme.lower() == "basic" and hmac.compare_digest(
        password.encode("utf-8"), ACCESS_CODE.encode("utf-8")
    )


@app.middleware("http")
async def require_access_code(request: Request, call_next):
    """Every page and API call needs the access code, if one is set. The
    401 makes the browser ask for it in its own popup; it then sends it
    by itself with every request. The chat WebSocket is checked in
    chat_session: HTTP middleware never sees it."""
    if _access_ok(request.headers):
        return await call_next(request)
    return Response(
        "access code required",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="bot-roulette", charset="UTF-8"'},
    )


def _entry_story(spec, when) -> str:
    """Generate this session's PAIRED entry story, pinned to `when`.

    Entry stories are never stored with the persona: a saved one would
    freeze the persona into re-living the moment it was generated on,
    and in spectator mode two personas would carry different clocks and
    argue about the time of day on air. One LLM call per persona per
    session; both spectated personas get the SAME `when`. The only copy
    that is kept is the one inside a saved chat (chats.py), so resuming
    that chat continues the same scene.
    """
    return generate_entry_story(llm, spec.core, when=when)


def _persona_turns(persona) -> int:
    """How many replies the persona has given in this chat, restored ones
    included: the starting value of a session's reply id (see
    chat_session)."""
    return sum(1 for m in persona.history if m["role"] == "assistant")


async def _edit_last_reply(
    ws: WebSocket, persona: Persona, data: dict, turn: int
) -> bool:
    """Apply the user's rewrite of the persona's LAST reply (1:1 chat).

    The persona remembers the edited version from then on: it replaces
    the reply in its history, in the model's own output format (see
    Persona.set_last_turn). Answers {"type": "edited", ...} with the
    turn as stored — exactly what a resumed chat will replay — or
    {"type": "edit_rejected", ...}. Returns True if the history changed.

    `turn` is the session's id for the last reply, and the edit must
    quote it: an edit that reaches the server after a newer reply was
    generated (or the last one regenerated) would otherwise rewrite the
    wrong reply. Fields left out keep their current value, the cap on
    typed messages applies, and [LEAVE] is stripped: leaving is the
    persona's own move, not something an edit can make it do.
    """
    last = persona.history[-1]
    if not turn or data.get("turn") != turn or last["role"] != "assistant":
        await ws.send_json({
            "type": "edit_rejected", "turn": data.get("turn"),
            "reason": "this reply can no longer be edited",
        })
        return False
    state, thoughts, _ = parse_response(last["content"])
    reply = LEAVE_RE.sub("", str(data.get("reply", "")))[:MAX_MESSAGE_CHARS]
    if not split_bubbles(reply):
        await ws.send_json({
            "type": "edit_rejected", "turn": turn,
            "reason": "the reply can't be empty",
        })
        return False
    persona.set_last_turn(
        str(data.get("state", state))[:MAX_MESSAGE_CHARS],
        str(data.get("thoughts", thoughts))[:MAX_MESSAGE_CHARS],
        reply,
    )
    state, thoughts, stored = parse_response(persona.history[-1]["content"])
    await ws.send_json({
        "type": "edited", "turn": turn, "state": state,
        "thoughts": thoughts, "bubbles": split_bubbles(stored),
    })
    return True


def _session_moment(first: dict) -> datetime:
    """The moment the entry story is built around.

    Prefer the CLIENT's local time (sent as a naive ISO string in the
    start message): the persona is texting this user, so the scene should
    match the time of day the user is actually living. The server clock
    is only a fallback — in production it's a UTC container, which would
    put the persona 1-2h behind the user (and on the wrong weekday
    around midnight).
    """
    raw = str(first.get("local_time", "")).strip()
    if raw:
        try:
            # Naive local time, e.g. "2026-07-25T22:30". We want the
            # wall-clock time as the user sees it, not an instant to be
            # converted, so drop any timezone designator instead of
            # honouring it: keeping it would make the value offset-aware
            # and break the comparison against the (naive) server clock.
            when = datetime.fromisoformat(raw)
            if when.tzinfo is not None:
                when = when.replace(tzinfo=None)
            # Sanity-check: a client with a wildly wrong clock (or a
            # hand-crafted payload) shouldn't set the scene in 1900.
            if abs(when - datetime.now()) <= timedelta(days=2):
                return when
            logger.warning(f"ignoring implausible client local_time: {raw!r}")
        except (ValueError, TypeError):
            logger.warning(f"ignoring malformed client local_time: {raw!r}")
    return datetime.now()


def find_listed_personas() -> list[Path]:
    """Enriched personas shown in the picker (chat-ready or not).

    Delete still needs to see people waiting on voice calibration;
    chat and spectate use ``find_chat_ready_personas``.
    """
    return find_enriched_personas(home.PERSONAS_DIR)


def check_generation_allowed() -> None:
    if not GENERATION_ALLOWED:
        raise HTTPException(
            status_code=403,
            detail="persona creation is disabled on this server",
        )


def check_cancellation_allowed() -> None:
    if not CANCELLATION_ALLOWED:
        raise HTTPException(
            status_code=403,
            detail="persona deletion is disabled on this server",
        )


NOT_CONFIGURED = "no model is set up yet — open ⚙ settings in the picker"


def check_configured() -> None:
    if not is_configured():
        raise HTTPException(status_code=503, detail=NOT_CONFIGURED)


# --- Persona creation jobs -------------------------------------------------
# The development pipeline (interview -> enrichment -> voice) takes
# minutes and many LLM calls. It runs in a plain thread; the client polls.
# Only ONE creation job may run at a time (don't stress the API); chatting
# stays fully available while a job runs. POST /jobs/{id}/cancel stops a
# job at its next model call (see llm.Cancelled); whatever it already
# wrote stays on disk and can be resumed with `bot-roulette enrich`.

JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

# How many finished (done/error/cancelled) jobs to keep around for
# post-mortem polling. Older finished jobs are pruned when a new job
# starts so the dict can't grow without bound over a long-running server.
JOBS_RETAIN_FINISHED = 10


def _prune_finished_jobs() -> None:
    """Drop oldest finished jobs beyond the retention cap.

    Must be called with _JOBS_LOCK held.
    """
    finished = [
        (jid, j) for jid, j in JOBS.items() if j["status"] != "running"
    ]
    # Most-recent-first by `ended` (fall back to `started` for safety).
    finished.sort(key=lambda item: item[1].get("ended", item[1]["started"]), reverse=True)
    for jid, _ in finished[JOBS_RETAIN_FINISHED:]:
        JOBS.pop(jid, None)


def _start_job(kind: str, fn) -> str:
    with _JOBS_LOCK:
        # Check running jobs inline: a helper that also takes _JOBS_LOCK
        # would deadlock (threading.Lock is not re-entrant).
        if any(job["status"] == "running" for job in JOBS.values()):
            raise HTTPException(
                status_code=409,
                detail="a persona is already being created — wait for it to finish",
            )
        _prune_finished_jobs()
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "kind": kind,
            "status": "running",
            "started": time.time(),
            "log": [],
            "cancel": threading.Event(),  # set by POST /jobs/{id}/cancel
        }

    def run():
        log = JOBS[job_id]["log"]
        # Every model call on this thread now honours the stop button.
        set_cancel_event(JOBS[job_id]["cancel"])
        # Capture the pipeline's logger output for THIS thread only:
        # logging from chat sessions / other threads stays out.
        ident = threading.get_ident()
        sink_id = logger.add(
            lambda msg: log.extend(str(msg).rstrip("\n").split("\n")),
            format="{message}",
            filter=lambda record: record["thread"].id == ident,
        )
        try:
            config_path = fn()
            with _JOBS_LOCK:
                JOBS[job_id].update(
                    status="done", persona=config_path.parent.name,
                    ended=time.time(),
                )
            # Keep the log with the persona for post-mortem reading.
            try:
                (config_path.parent / "generation_log.txt").write_text(
                    "\n".join(log) + "\n", encoding="utf-8"
                )
            except OSError:
                pass
        except Cancelled:
            # A BaseException, so it needs its own clause: unhandled, it
            # would kill this thread and leave the job "running" forever.
            logger.warning("persona creation stopped by the user")
            with _JOBS_LOCK:
                JOBS[job_id].update(status="cancelled", ended=time.time())
        except Exception as e:
            # The pipeline log (captured above) already has the trace;
            # log it at error level too so it shows in the console even
            # if no one is polling the job.
            logger.exception("persona creation job failed")
            with _JOBS_LOCK:
                JOBS[job_id].update(
                    status="error", error=str(e), ended=time.time()
                )
        finally:
            set_cancel_event(None)
            logger.remove(sink_id)

    threading.Thread(target=run, daemon=True).start()
    return job_id


@app.get("/")
async def index():
    # no-store: the client is a single file; never let the browser run a
    # stale copy against a newer server protocol.
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/personas")
async def list_personas():
    """Enriched personas for the picker: id (folder name), display name,
    core preview, whether it is chat-ready (required before chatting),
    and whether its voice was calibrated (a chat-ready persona may have
    had no one to calibrate against)."""
    personas = []
    for cfg in sorted(find_listed_personas()):
        try:
            spec = load_persona(cfg)
        except Exception:
            continue
        personas.append(
            {
                "id": cfg.parent.name,
                "name": spec.name,
                "core": spec.core,
                "chat_ready": spec.chat_ready,
                "voice_calibrated": spec.voice_calibrated,
            }
        )
    personas.sort(key=lambda p: p["name"].lower())
    return {
        "personas": personas,
        "generation_allowed": GENERATION_ALLOWED,
        "cancellation_allowed": CANCELLATION_ALLOWED,
        "memory_allowed": CHAT_MEMORY_ALLOWED,
        "configured": is_configured(),
    }


@app.get("/personas/generation-options")
async def generation_options():
    """The background lists offered by the generation form."""
    return CONFIG.random_personas.model_dump()


@app.get("/personas/{persona_id}/chats")
async def list_saved_chats(persona_id: str):
    """Saved chats for ONE persona (most recent first), for the resume list.

    Chats are only ever listed per-persona; there is no cross-persona
    listing, and a chat's content is only returned over the websocket
    that resumes it by id.
    """
    if not CHAT_MEMORY_ALLOWED:
        raise HTTPException(
            status_code=403, detail="chat memory is disabled on this server"
        )
    from . import chats

    return {"chats": chats.list_chats(persona_id)}


@app.delete("/personas/{persona_id}/chats/{chat_id}")
async def delete_saved_chat(persona_id: str, chat_id: str):
    """Permanently delete one saved chat (not the persona)."""
    if not CHAT_MEMORY_ALLOWED:
        raise HTTPException(
            status_code=403, detail="chat memory is disabled on this server"
        )
    from . import chats

    try:
        removed = chats.delete_chat(persona_id, chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid chat id")
    if not removed:
        raise HTTPException(status_code=404, detail="unknown chat")
    return {"deleted": chat_id}


@app.post("/personas/generate")
async def generate_persona(payload: dict | None = Body(default=None)):
    """Invent a persona around the given identity and develop it
    (background job)."""
    check_generation_allowed()
    check_configured()
    # Validated here, after the checks above, so that a server that can't
    # generate says so whatever the request.
    try:
        request = GeneratePersonaRequest.model_validate(payload or {})
        fixed_traits = validate_fixed_seed_traits(
            request.fixed_traits, CONFIG.random_personas
        )
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: "
            f"{err['msg'].removeprefix('Value error, ')}"
            for err in e.errors()
        )
        raise HTTPException(status_code=400, detail=problems) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    job_id = _start_job(
        "generate",
        lambda: build_random_persona(
            llm,
            identity=request.identity,
            root=home.PERSONAS_DIR,
            creation=CONFIG.creation,
            random_personas=CONFIG.random_personas,
            fixed_traits=fixed_traits,
        ),
    )
    return {"job": job_id}


@app.post("/personas/import-cv")
async def import_cv(pdf: UploadFile = File(...)):
    """Build a persona from an uploaded CV PDF (background job)."""
    check_generation_allowed()
    check_configured()

    # Read at most one byte past the limit: an oversized upload must be
    # rejected without first loading all of it into memory.
    data = await pdf.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF too large (max {MAX_PDF_MB} MB)",
        )
    # Reject non-PDF uploads up front: a wrong file type would otherwise
    # fail deep inside the background pipeline and only surface as a
    # generic "error" status after the client has been polling.
    if not data[:5] == b"%PDF-":
        raise HTTPException(
            status_code=415,
            detail="upload must be a PDF (missing %PDF- header)",
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(data)
    tmp.close()

    def run():
        try:
            return build_persona_from_cv(
                llm, tmp.name, root=home.PERSONAS_DIR, creation=CONFIG.creation
            )
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    try:
        job_id = _start_job("import-cv", run)
    except Exception:
        # If another creation job won the race, `run` is never started and
        # therefore cannot execute its cleanup block.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return {"job": job_id}


@app.delete("/personas/{persona_id}")
async def delete_persona(persona_id: str):
    """Permanently delete a persona folder (the developed persona and its
    root v0.1 parent, if present) and its saved chats."""
    check_cancellation_allowed()

    import shutil

    # persona_id is the folder name of an enriched persona (voice-
    # calibrated or still waiting). Resolve it against that set so
    # callers can't delete arbitrary folders on disk.
    candidates = {cfg.parent.name: cfg for cfg in find_listed_personas()}
    cfg = candidates.get(persona_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="unknown persona")

    target = cfg.parent

    # Also remove the root v0.1 parent folder if this is an enriched
    # persona (its YAML records `parent`, which is the root folder name).
    try:
        spec = load_persona(cfg)
        parent_name = getattr(spec, "parent", None)
    except Exception:
        parent_name = None
    root_folder = None
    if parent_name and parent_name != "none":
        # `parent` is data from YAML and must never widen what this
        # endpoint can delete. Compare RESOLVED paths: a name check alone
        # let ".." (the folder containing the project) and, on Windows,
        # "..." (the project itself) through. And the folder must really
        # be a v0.1 root persona — not just any sibling (.git, static).
        candidate_root = (target.parent / str(parent_name)).resolve()
        if (
            candidate_root.parent == target.parent.resolve()
            and candidate_root != target.resolve()
            and candidate_root.is_dir()
        ):
            try:
                if load_persona(candidate_root / cfg.name).parent == "none":
                    root_folder = candidate_root
            except Exception:
                pass

    # Keep the root folder if anything else still depends on it: another
    # enriched persona sharing the same parent, or the root itself being
    # a listed chat-ready persona. Deleting it would silently destroy a
    # persona the user didn't ask to remove.
    if root_folder is not None:
        for other_id, other_cfg in candidates.items():
            if other_id == persona_id:
                continue
            if other_cfg.parent.resolve() == root_folder:
                root_folder = None  # the root is itself chat-ready
                break
            try:
                other_parent = load_persona(other_cfg).parent
            except Exception:
                continue
            if other_parent == parent_name:
                root_folder = None  # a sibling still references it
                break

    shutil.rmtree(target)
    if root_folder is not None:
        shutil.rmtree(root_folder)

    # Its saved chats go too: with the persona gone they can never be
    # listed or resumed again, so they would only linger on disk.
    try:
        from . import chats

        chats.delete_persona_chats(persona_id)
    except Exception:
        logger.exception("failed to delete the persona's saved chats")

    return {"deleted": persona_id}


@app.get("/jobs/active")
async def active_job():
    """The currently running creation job, if any — lets a freshly loaded
    client resume the busy state and its polling."""
    # One pass under the lock: finding the id and reading its kind in two
    # separate critical sections could see the job finish in between.
    with _JOBS_LOCK:
        for job_id, job in JOBS.items():
            if job["status"] == "running":
                return {"job": job_id, "kind": job["kind"]}
    return {"job": None, "kind": None}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, since: int = Query(0)):
    """Job status plus incremental log: `since` is the number of log
    lines the client already has; only newer lines are returned."""
    # Snapshot everything under the lock: the pipeline thread updates
    # status/persona/error/ended concurrently.
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        log = job.get("log", [])
        # Snapshot the length BEFORE slicing: the pipeline thread appends
        # concurrently, and a line added between the slice and len(log)
        # would be counted in log_total without ever being sent — the
        # client would then skip past it forever.
        n = len(log)
        since = max(0, min(since, n))
        # Freeze elapsed once the job has finished: a finished job's
        # duration shouldn't keep growing as the client polls.
        end = job.get("ended")
        elapsed = int((end or time.time()) - job["started"])
        out = {"kind": job["kind"], "status": job["status"],
               "elapsed": elapsed,
               "log": list(log[since:n]), "log_total": n}
        if "persona" in job:
            out["persona"] = job["persona"]
        if "error" in job:
            out["error"] = job["error"]
        if job["status"] == "running" and job["cancel"].is_set():
            out["cancelling"] = True
    return out


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Stop a running creation job. The model call in flight can't be
    interrupted, so the job stops at its NEXT call: until then its status
    stays "running" with "cancelling": true, then becomes "cancelled"."""
    check_generation_allowed()
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job["status"] != "running":
            raise HTTPException(
                status_code=409, detail=f"the job is already {job['status']}"
            )
        job["cancel"].set()
    return {"job": job_id, "status": "cancelling"}


# --- Model and API key (the ⚙ settings panel) ------------------------------
# The key never leaves the server: GET shows a hint of it, POST replaces it.

_SETTINGS_SAVE = asyncio.Lock()  # one save at a time: check + write as a unit


class SettingsUpdate(BaseModel):
    server: str | None = Field(None, max_length=300)  # None = where it runs now
    model: str = Field("", max_length=200)
    api_key: str = Field("", max_length=200)


async def _settings_view() -> dict:
    """settings_summary() plus "model_known": whether the model is one
    OpenRouter's catalog has, or one your own server lists (None when no
    model is set or the server can't be asked). A model set elsewhere —
    by hand in .env, or a stray OS variable — may be one the server
    doesn't have; the panel says so."""
    view = settings_summary()
    view["model_known"] = None
    if view["model"]:
        try:
            view["model_known"] = await asyncio.to_thread(_is_known_model, view["model"])
        except (OpenRouterUnavailable, ServerUnavailable, ValueError):
            pass
    return view


def _is_known_model(model: str) -> bool:
    url = server_url()
    return is_served_model(url, model, server_api_key()) if url else is_known_model(model)


def _check_settings(server: str, model: str, api_key: str, moving: bool) -> None:
    """ValueError unless ``server`` (your server's address, or "" for
    OpenRouter) accepts the new key and has the model; an empty value is
    left unchecked. A saved key is only sent to the server it was saved
    for: on a new address, the check uses the new key or none."""
    if not server:
        if api_key:
            check_api_key(api_key)
        if model:
            check_model(model)
        return
    key = api_key or ("" if moving else server_api_key())
    if model:
        # Your server lists its models only for a key it accepts: one
        # call checks both.
        check_server_model(server, model, key)
    else:
        server_models(server, key)


@app.get("/settings")
async def get_settings():
    """Where the model runs, the model and hints of the API keys (their
    last 4 characters)."""
    return await _settings_view()


@app.get("/settings/models")
async def list_models(server: str | None = Query(None, max_length=300)):
    """The panel's suggestions: OpenRouter's catalog, or the models a
    server lists. ``server`` asks about an address being typed in the
    panel ("" for OpenRouter); left out, about where the model runs now.
    A saved key only goes to the server it was saved for."""
    current = server_url()
    url = current if server is None else server.strip().rstrip("/")
    try:
        if url:
            key = server_api_key() if url == current else ""
            ids = await asyncio.to_thread(server_models, url, key)
            return {"models": [{"id": i, "name": i} for i in ids]}
        return {"models": await asyncio.to_thread(model_catalog)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (OpenRouterUnavailable, ServerUnavailable) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/settings")
async def update_settings(update: SettingsUpdate):
    """Change where the model runs, the model and/or the API key.

    ``server`` is your server's address, or "" for OpenRouter; left out,
    the model stays where it runs. An empty model or key, or the model as
    it already is, keeps its current value. Moving to another server
    needs a model it has (the current one, if it has it) and, on
    OpenRouter, a key. Every changed value is checked first, with
    OpenRouter or the server (free calls: no tokens), and nothing is
    saved unless all of them pass — a typo must not break every chat.
    Then everything is written to .env and applied at once.
    """
    async with _SETTINGS_SAVE:
        current = settings_summary()
        server = current["server"]
        if update.server is not None:
            server = update.server.strip().rstrip("/")
        moving = server != current["server"]
        model = update.model.strip()
        if model == current["model"] and not moving:
            model = ""
        api_key = update.api_key.strip()
        if not (moving or model or api_key):
            raise HTTPException(status_code=400, detail="nothing to change")
        checked_model = model or (current["model"] if moving else "")
        if moving and not checked_model:
            raise HTTPException(status_code=400, detail="choose the model to use there")
        if moving and not server and not (api_key or current["openrouter_key_hint"]):
            raise HTTPException(status_code=400, detail="OpenRouter needs your API key")
        try:
            await asyncio.to_thread(_check_settings, server, checked_model, api_key, moving)
            await asyncio.to_thread(
                save_settings,
                server=server if moving else None,
                model=model or None,
                api_key=api_key or None,
                llm=llm,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (OpenRouterUnavailable, ServerUnavailable) as e:
            raise HTTPException(status_code=502, detail=f"{e} — nothing was saved")
        except OSError:
            logger.exception("failed to write the settings to .env")
            raise HTTPException(
                status_code=500, detail="couldn't write .env on the server"
            )
    summary = settings_summary()
    changed = (
        ([f"server -> {summary['server'] or 'OpenRouter'}"] if moving else [])
        + ([f"model -> {summary['model']}"] if model else [])
        + ([f"API key -> {summary['api_key_hint']}"] if api_key else [])
    )
    logger.info(f"settings saved: {', '.join(changed)}")
    return await _settings_view()


@app.websocket("/ws")
async def chat_session(ws: WebSocket):
    await ws.accept()
    # The browser sends the access code with the WebSocket too, as with
    # every request once it has asked for it.
    if not _access_ok(ws.headers):
        await ws.send_json(
            {"type": "ended", "reason": "access code required — reload the page"}
        )
        await ws.close()
        return

    # The first message starts the session: {"type": "start", ...}. Short
    # timeout so idle sockets don't linger.
    try:
        first = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except Exception:
        await ws.close()
        return
    if not isinstance(first, dict) or first.get("type") != "start":
        await ws.close()
        return

    # The start message names the chosen persona (its folder name).
    wanted = str(first.get("persona", "")).strip()
    listed = {cfg.parent.name: cfg for cfg in find_listed_personas()}
    ready = {cfg.parent.name: cfg for cfg in find_chat_ready_personas(home.PERSONAS_DIR)}
    if wanted not in listed:
        await ws.send_json({"type": "ended", "reason": "unknown persona"})
        await ws.close()
        return
    if wanted not in ready:
        await ws.send_json({
            "type": "ended",
            "reason": "persona is not ready yet (voice calibration pending)",
        })
        await ws.close()
        return
    config_path = ready[wanted]

    # Spectator mode: the start message names a SECOND persona; the two
    # talk to each other and the user just watches.
    wanted_b = str(first.get("persona_b", "")).strip()
    if wanted_b:
        if wanted_b not in listed:
            await ws.send_json({"type": "ended", "reason": "unknown persona"})
            await ws.close()
            return
        if wanted_b not in ready:
            await ws.send_json({
                "type": "ended",
                "reason": "persona is not ready yet (voice calibration pending)",
            })
            await ws.close()
            return
        if not is_configured():
            await ws.send_json({"type": "ended", "reason": NOT_CONFIGURED})
            await ws.close()
            return
        await _spectate(ws, config_path, ready[wanted_b], _session_moment(first))
        return

    # Chat memory: the start message may name a saved chat to RESUME.
    # Resuming means: the persona gets a FRESH system prompt (built from
    # the persona's current files and the entry story saved with the
    # chat) plus exactly the turns saved in that one chat file — and
    # nothing else. No other chat's content is ever loaded, so histories
    # can't leak between chats.
    chat_id = str(first.get("chat_id", "")).strip() or None
    saved = None
    if CHAT_MEMORY_ALLOWED:
        from . import chats

        if chat_id:
            try:
                saved = chats.load_chat(wanted, chat_id)
            except ValueError:
                saved = None
            if saved is None:
                await ws.send_json({"type": "ended", "reason": "unknown chat"})
                await ws.close()
                return
        else:
            chat_id = chats.new_chat_id()

    # Every chat needs the model, except an ended one: that is only
    # replayed, read-only, and stays viewable without a key.
    if not (saved and saved.get("ended")) and not is_configured():
        await ws.send_json({"type": "ended", "reason": NOT_CONFIGURED})
        await ws.close()
        return

    # The PAIRED entry story. A new chat gets one built around the user's
    # local time — an LLM call, so run it off the event loop and show a
    # typing indicator meanwhile. A resumed chat reuses the story saved
    # with it: the scenario lives in the system prompt, which is never
    # saved, so without it the persona would lose the situation, the
    # stranger rules and the leave button. Chats saved before stories
    # were stored get a fresh one — unless they're read-only anyway.
    await ws.send_json({"type": "typing"})
    try:
        spec = load_persona(config_path)
        entry_story = str((saved or {}).get("entry_story") or "").strip()
        if not entry_story and not (saved and saved.get("ended")):
            entry_story = await asyncio.to_thread(
                _entry_story, spec, _session_moment(first)
            )
        persona = Persona(
            llm,
            config_path=config_path,
            scenario_text=paired_scenario(entry_story) if entry_story else "",
            use_judge=JUDGE_WEB_CHATS,
        )
        if saved is not None:
            persona.restore_history(saved["messages"])
    except Exception:
        logger.exception("failed to set up the chat session")
        await ws.send_json(
            {"type": "ended", "reason": "couldn't start the chat — try again"}
        )
        await ws.close()
        return

    # Id of the persona's latest reply, quoted back by "edit" and
    # "regenerate". It changes with EVERY new reply, including one that
    # replaces the last (regenerate leaves the reply count unchanged), so
    # an action meant for an older reply can never land on a newer one.
    reply_no = _persona_turns(persona)

    await ws.send_json({"type": "matched", "name": persona.spec.name})
    await ws.send_json(
        {
            "type": "core",
            "name": persona.spec.name,
            "text": persona.spec.core,
            "scenario": persona.spec.scenarios.get("PAIRED", ""),
        }
    )
    if CHAT_MEMORY_ALLOWED and chat_id:
        # Replay the saved turns so the client renders the conversation
        # exactly as it was, and tell it the id so a new chat's id is
        # known for the "resume later" list. The "chat" message is sent
        # AFTER the replay: the client renders its "earlier messages
        # restored" divider where it arrives, so sending it first would
        # put the divider above the restored history instead of below.
        ended_saved = saved is not None and saved.get("ended")
        persona_left = False
        if saved is not None:
            for m in saved["messages"]:
                if m["role"] == "user":
                    await ws.send_json({"type": "replay", "who": "me", "text": m["content"]})
                else:
                    # Saved assistant turns are the raw <state>/<thoughts>/<reply>
                    # model output: replay the inner monologue first, then the
                    # visible reply bubbles, split exactly as they were live.
                    state, thoughts, reply = parse_response(m["content"])
                    if state or thoughts:
                        await ws.send_json(
                            {"type": "replay", "who": "inner",
                             "state": state, "thoughts": thoughts}
                        )
                    for b in split_bubbles(reply):
                        await ws.send_json(
                            {"type": "replay", "who": "them", "text": b}
                        )
                    # The persona's LAST turn tells how an ended chat
                    # ended: [LEAVE] there means it left in-fiction.
                    persona_left = bool(LEAVE_RE.search(reply))
        await ws.send_json({
            "type": "chat", "id": chat_id, "resumed": saved is not None,
            # A resumed chat's last reply is editable too, until the next
            # message: the client needs its number to quote in an edit.
            "turn": reply_no,
        })
        if ended_saved:
            # Read-only viewing: the chat stays archived — full history
            # replayed above, but no new messages are accepted. It ended
            # because the persona left in-fiction, or because the session
            # limit was reached; say which.
            reason = (
                f"{persona.spec.name} left this chat"
                if persona_left
                else "this chat reached the session limit"
            )
            await ws.send_json(
                {"type": "ended", "reason": f"{reason} — it's kept read-only"}
            )
            await ws.close()
            return

    # Restored turns count against the session limit: starting at zero
    # here would let a resumed chat grow its prompt (and the bill)
    # without bound across successive resumes.
    messages_used = (
        sum(1 for m in saved["messages"] if m["role"] == "user") if saved else 0
    )

    def save(ended: bool = False) -> None:
        """Persist the chat (a no-op without chat memory)."""
        if not (CHAT_MEMORY_ALLOWED and chat_id):
            return
        try:
            from . import chats

            chats.save_chat(
                wanted, chat_id, persona.history,
                entry_story=entry_story, ended=ended,
            )
        except Exception:
            logger.exception("failed to save chat")

    async def deliver(answer) -> bool:
        """Send a new reply — to a message, or regenerated — and persist
        the chat. Returns True if the persona left (the session is over)."""
        nonlocal reply_no
        reply_no += 1
        # Inner monologue, shown out-of-band in the client.
        await ws.send_json(
            {"type": "inner", "state": answer.state, "thoughts": answer.thoughts}
        )
        bubbles = [b for b in answer.reply.split("\n") if b.strip()]
        for bubble in bubbles:
            await ws.send_json({"type": "bubble", "text": bubble})
        if answer.left:
            save(ended=True)
            await ws.send_json(
                {"type": "ended", "reason": f"{persona.spec.name} left the chat"}
            )
            return True
        # Persist after every completed turn: the chat is resumable
        # even if the server restarts or the tab just closes.
        save()
        await ws.send_json({"type": "done", "turn": reply_no})
        return False

    try:
        while True:
            data = await ws.receive_json()
            if not isinstance(data, dict):
                continue
            if data.get("type") == "edit":
                if await _edit_last_reply(ws, persona, data, reply_no):
                    # Saved at once: a resume must bring back what the
                    # persona now remembers, not the original reply.
                    save()
                continue
            if data.get("type") == "regenerate":
                # A fresh reply to the same message, replacing the last
                # one. Same staleness rule as "edit" (see reply_no). Not
                # capped and not counted as a message: it replaces a
                # reply instead of adding one, so the chat doesn't grow.
                if (
                    not reply_no
                    or data.get("turn") != reply_no
                    or not persona.can_regenerate
                ):
                    await ws.send_json({
                        "type": "regenerate_rejected", "turn": data.get("turn"),
                        "reason": "this reply can no longer be regenerated",
                    })
                    continue
                await ws.send_json({"type": "typing"})
                try:
                    answer = await asyncio.to_thread(persona.regenerate)
                except Exception:
                    # Unlike a failed reply, this doesn't end the chat:
                    # Persona.regenerate put the previous reply back.
                    logger.exception("regeneration failed")
                    await ws.send_json({
                        "type": "regenerate_rejected", "turn": reply_no,
                        "reason": "generation failed, so the previous reply was kept",
                    })
                    continue
                # The client drops the replaced reply; the new one follows
                # as usual (inner, bubbles, done).
                await ws.send_json({"type": "regenerated", "turn": reply_no})
                if await deliver(answer):
                    break
                continue
            if data.get("type") != "message":
                continue
            text = str(data.get("text", "")).strip()
            if not text:
                continue
            text = text[:MAX_MESSAGE_CHARS]

            messages_used += 1
            if messages_used > MAX_MESSAGES_PER_SESSION:
                save(ended=True)
                await ws.send_json(
                    {"type": "ended", "reason": "session limit reached"}
                )
                break

            await ws.send_json({"type": "typing"})
            # The OpenAI client is synchronous; don't block the event loop.
            try:
                answer = await asyncio.to_thread(persona, text)
            except Exception:
                # The real cause is logged server-side for debugging; the
                # user-facing reason is honest but generic so it doesn't
                # leak implementation details or break the persona fiction.
                logger.exception("persona reply failed")
                await ws.send_json(
                    {"type": "ended", "reason": "something went wrong on our end"}
                )
                break
            if await deliver(answer):
                break
    except (WebSocketDisconnect, RuntimeError):
        # WebSocketDisconnect: the user left mid-session. RuntimeError:
        # the socket closed while a reply was being generated, so a
        # later send hit a closed socket. The history was already saved
        # turn-by-turn above, so the chat can simply be resumed later.
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


async def _spectate(
    ws: WebSocket, config_a: Path, config_b: Path, now: datetime
) -> None:
    """Run a conversation between two personas while the user watches.

    The two personas are independent `Persona` instances (each with its
    own PAIRED scenario and history); each one's reply is fed to the
    other as the next user message. The spectator can stop at any time
    with {"type": "stop"} or by closing the socket.
    """
    try:
        # ONE moment for both personas (the spectator's local time):
        # independently generated entry stories used to disagree about
        # the time of day (one at 16:40 at her desk, the other at 21:40
        # on the sofa), and the personas would then argue about it on
        # air. Same timestamp = same clock.
        spec_a = load_persona(config_a)
        spec_b = load_persona(config_b)
        await ws.send_json({"type": "typing"})
        story_a, story_b = await asyncio.gather(
            asyncio.to_thread(_entry_story, spec_a, now),
            asyncio.to_thread(_entry_story, spec_b, now),
        )
        persona_a = Persona(
            llm, config_path=config_a, scenario_text=paired_scenario(story_a),
            use_judge=JUDGE_WEB_CHATS,
        )
        persona_b = Persona(
            llm, config_path=config_b, scenario_text=paired_scenario(story_b),
            use_judge=JUDGE_WEB_CHATS,
        )
    except Exception:
        logger.exception("failed to load spectate personas")
        await ws.send_json({"type": "ended", "reason": "could not load personas"})
        await ws.close()
        return

    await ws.send_json(
        {
            "type": "matched",
            "mode": "spectate",
            "name": persona_a.spec.name,
            "name_b": persona_b.spec.name,
        }
    )
    # Combined "who are they?" panel: both cores, both scenarios.
    await ws.send_json(
        {
            "type": "core",
            "name": f"{persona_a.spec.name} & {persona_b.spec.name}",
            "text": (
                f"=== {persona_a.spec.name} ===\n{persona_a.spec.core}\n\n"
                f"=== {persona_b.spec.name} ===\n{persona_b.spec.core}"
            ),
            "scenario": (
                f"=== {persona_a.spec.name} ===\n"
                f"{persona_a.spec.scenarios.get('PAIRED', '')}\n\n"
                f"=== {persona_b.spec.name} ===\n"
                f"{persona_b.spec.scenarios.get('PAIRED', '')}"
            ),
        }
    )

    # Listen for control messages from the spectator while the
    # generation loop drives the conversation. Supported messages:
    #   {"type": "stop"}                      end the show
    #   {"type": "continue"}                  accept the last turn as-is
    #   {"type": "edit", "who": a|b,          edit the last turn before
    #    "state": ..., "thoughts": ...,       it is fed to the other persona
    #    "reply": ...}
    stopped = asyncio.Event()
    pending: asyncio.Queue = asyncio.Queue()

    async def listen():
        try:
            while True:
                data = await ws.receive_json()
                t = data.get("type")
                if t == "stop":
                    stopped.set()
                    return
                elif t in ("continue", "edit"):
                    await pending.put(data)
        except Exception:  # disconnect or bad frame: stop generating
            stopped.set()

    listener = asyncio.create_task(listen())

    async def await_control() -> dict | None:
        """Wait for the spectator's decision on the just-shown turn.
        Returns the control message (continue/edit) or None on stop."""
        get_task = asyncio.create_task(pending.get())
        stop_task = asyncio.create_task(stopped.wait())
        await asyncio.wait(
            [get_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stopped.is_set():
            get_task.cancel()
            stop_task.cancel()
            return None
        # The pending.get() task won; cancel the stop watcher.
        stop_task.cancel()
        try:
            return get_task.result()
        # CancelledError is a BaseException (not Exception) since 3.8,
        # so it must be listed explicitly.
        except (asyncio.CancelledError, Exception):
            return None

    try:
        speaker, who = persona_a, "a"
        text = SPECTATE_OPENING_NUDGE
        for _ in range(SPECTATE_MAX_TURNS):
            if stopped.is_set():
                await ws.send_json(
                    {"type": "ended", "reason": "you stopped watching"}
                )
                return

            await ws.send_json({"type": "typing", "name": speaker.spec.name})
            # The OpenAI client is synchronous; don't block the event loop.
            try:
                answer = await asyncio.to_thread(speaker, text)
            except Exception:
                logger.exception("spectate persona reply failed")
                await ws.send_json(
                    {"type": "ended", "reason": "something went wrong on our end"}
                )
                return
            if stopped.is_set():
                await ws.send_json(
                    {"type": "ended", "reason": "you stopped watching"}
                )
                return

            await ws.send_json(
                {
                    "type": "inner",
                    "who": who,
                    "name": speaker.spec.name,
                    "state": answer.state,
                    "thoughts": answer.thoughts,
                }
            )
            bubbles = [b for b in answer.reply.split("\n") if b.strip()]
            for bubble in bubbles:
                await ws.send_json(
                    {
                        "type": "bubble",
                        "who": who,
                        "name": speaker.spec.name,
                        "text": bubble,
                    }
                )

            if answer.left:
                await ws.send_json(
                    {
                        "type": "ended",
                        "reason": f"{speaker.spec.name} left the chat",
                    }
                )
                return

            # --- Edit window ------------------------------------------
            # The spectator can edit the just-shown turn (state,
            # thoughts, reply) before it's fed to the other persona.
            # The server pauses here until the spectator sends
            # "continue" (accept as-is) or "edit" (rewrite the turn).
            # Drain stale control messages first: a leftover continue/
            # edit from a previous turn (double-clicks, fast mode
            # toggling) must never be consumed for THIS turn — applying
            # an old edit to the wrong turn corrupts the persona's
            # history and derails the conversation.
            while not pending.empty():
                pending.get_nowait()
            await ws.send_json(
                {
                    "type": "editable",
                    "who": who,
                    "name": speaker.spec.name,
                    "state": answer.state,
                    "thoughts": answer.thoughts,
                    "reply": answer.reply,
                }
            )
            ctrl = await await_control()
            if ctrl is None:  # spectator stopped / disconnected
                await ws.send_json(
                    {"type": "ended", "reason": "you stopped watching"}
                )
                return
            # Apply the edit only if it targets the CURRENT speaker; a
            # mismatched "who" means a stale message — treat as continue.
            if ctrl.get("type") == "edit" and str(ctrl.get("who", who)) == who:
                # Same rules as the 1:1 chat's edit (_edit_last_reply):
                # the cap on typed messages applies, and [LEAVE] is
                # stripped — leaving is the persona's own move.
                edited_state = str(ctrl.get("state", answer.state))[:MAX_MESSAGE_CHARS]
                edited_thoughts = str(ctrl.get("thoughts", answer.thoughts))[:MAX_MESSAGE_CHARS]
                edited_reply = LEAVE_RE.sub(
                    "", str(ctrl.get("reply", answer.reply))
                )[:MAX_MESSAGE_CHARS]
                # A reply left with no bubbles (emptied, or only markers)
                # would feed the other persona an empty message; keep the
                # original instead. Otherwise hand it on the way an
                # unedited reply goes: one bubble per line.
                edited_bubbles = split_bubbles(edited_reply)
                edited_reply = (
                    "\n".join(edited_bubbles) if edited_bubbles else answer.reply
                )
                speaker.set_last_turn(
                    edited_state, edited_thoughts, edited_reply
                )
                answer = answer.model_copy(
                    update={
                        "state": edited_state,
                        "thoughts": edited_thoughts,
                        "reply": edited_reply,
                    }
                )

            # Hand the reply to the other persona.
            text = answer.reply
            speaker, who = (
                (persona_b, "b") if who == "a" else (persona_a, "a")
            )

        await ws.send_json(
            {"type": "ended", "reason": "conversation limit reached"}
        )
    except (WebSocketDisconnect, RuntimeError):
        pass  # spectator left mid-show; sends to a closed socket raise
    finally:
        listener.cancel()
        try:
            await ws.close()
        except Exception:
            pass
