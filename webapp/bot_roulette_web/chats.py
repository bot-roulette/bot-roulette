"""Saved chat sessions: each chat is a self-contained history, stored as
one JSON file, resumed only by its own id.

Design notes:
- Isolation is structural: a chat file is ONLY ever read when its own
  id is supplied. Nothing ever reads another persona's folder, and no
  chat ever reads another chat's file — so there is no path by which
  one chat's content can leak into another's context.
- Chats live OUTSIDE the persona folders: those folders are deleted by
  the persona-deletion flow and duplicated by the enrichment pipeline;
  keeping chats in a separate tree keeps both those operations safe.
  Deleting a persona also deletes its chats (delete_persona_chats):
  they can never be listed or resumed once the persona is gone.
- One file per chat ({chat_id}.json), atomic writes (tmp + replace) so
  a crash mid-save can never corrupt a history someone is resuming.
- Each record keeps the chat's own PAIRED entry story. It exists
  nowhere else (persona folders never store one), and without it a
  resumed chat would lose its whole scenario: the situation, the
  stranger rules and the leave button.
- The whole module is gated by server.CHAT_MEMORY_ALLOWED: when memory
  is off, the server never reads or writes a chat and no chat file is
  ever created. The one call made either way is cleanup: deleting a
  persona removes any chats left from when memory was on.
"""

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from .home import CHATS_DIR

# Writes from concurrent sessions (different sockets, possibly the same
# chat reopened in two tabs) are serialized by this lock, so two saves
# can never interleave their tmp-file replace. Note: saves do NOT merge —
# each one overwrites the whole record, so the last writer wins. That is
# deliberate: a chat is single-address, and merging two divergent
# histories would invent a conversation nobody had.
_LOCK = threading.Lock()

# Only these roles are ever persisted: the system prompt is rebuilt on
# resume (never stored — see below) and "inner" fields like the judge
# verdict are per-turn metadata, not history.
_PERSIST_ROLES = ("user", "assistant")


def _chat_path(persona_id: str, chat_id: str) -> Path:
    # Both ids are data from the client/protocol. Only accept a plain
    # folder/file name; a crafted id must never widen this to an
    # arbitrary path on disk (same policy as persona deletion).
    # NOTE: "." and ".." pass the naive name check (Path("..").name IS
    # ".."), so they must be rejected explicitly, and the final path is
    # verified to stay inside CHATS_DIR as a last line of defence.
    for value in (persona_id, chat_id):
        value = str(value)
        p = Path(value)
        if (
            p.name != value
            or p.is_absolute()
            or not value
            or value in (".", "..")
        ):
            raise ValueError(f"unsafe id: {value!r}")
    path = CHATS_DIR / persona_id / f"{chat_id}.json"
    if CHATS_DIR.resolve() not in path.resolve().parents:
        raise ValueError(f"unsafe id: {persona_id!r}/{chat_id!r}")
    return path


def new_chat_id() -> str:
    return uuid.uuid4().hex[:12]


def _read_record(path: Path) -> dict | None:
    """A chat file's record, or None if it is unreadable or malformed.

    Valid JSON is not enough: a file holding a list, or messages that
    aren't {"role", "content"} dicts, used to raise AttributeError and
    take down the whole resume list (and the resume itself). Messages
    are filtered to well-formed persisted turns so callers can index
    m["role"] / m["content"] safely; timestamps (sort keys) are coerced
    to numbers and the entry story to a string.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return None
    data["messages"] = [
        m
        for m in data["messages"]
        if isinstance(m, dict)
        and m.get("role") in _PERSIST_ROLES
        and isinstance(m.get("content"), str)
    ]
    for key in ("created", "updated"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            data[key] = 0
    if not isinstance(data.get("entry_story"), str):
        data["entry_story"] = ""
    return data


def list_chats(persona_id: str) -> list[dict]:
    """All saved chats for ONE persona, most recent first.

    Returns id, turn count, preview of the last user message, and
    timestamps — enough for the picker to show a resume list without
    loading any history into memory.
    """
    try:
        folder = _chat_path(persona_id, "x").parent  # validated id
    except ValueError:
        return []
    if not folder.is_dir():
        return []
    out = []
    for f in folder.glob("*.json"):
        data = _read_record(f)
        if data is None:
            continue  # a corrupt file hides that one chat, not the list
        msgs = data["messages"]
        last_user = next(
            (m["content"] for m in reversed(msgs) if m["role"] == "user"), ""
        )
        out.append(
            {
                # The file name is the chat's address (see load_chat).
                "id": f.stem,
                "persona": persona_id,
                "created": data["created"],
                "updated": data["updated"],
                "turns": len(msgs) // 2,
                "preview": " ".join(last_user.split())[:80],
                "ended": bool(data.get("ended")),
            }
        )
    out.sort(key=lambda c: c["updated"], reverse=True)
    return out


def load_chat(persona_id: str, chat_id: str) -> dict | None:
    """One chat's full record, or None if it doesn't exist or is malformed.

    This is the ONLY way history ever leaves disk, and it is strictly
    addressed by (persona_id, chat_id): resume means the server hands
    these exact messages to the new Persona instance, and nothing else.
    """
    path = _chat_path(persona_id, chat_id)
    if not path.is_file():
        return None
    return _read_record(path)


def save_chat(
    persona_id: str,
    chat_id: str,
    messages: list[dict],
    *,
    entry_story: str = "",
    ended: bool = False,
    now: float | None = None,
) -> None:
    """Persist a chat's history.

    `messages` is the persona's full in-memory history INCLUDING the
    system prompt at index 0; the system prompt is dropped before
    writing. It is rebuilt from the (possibly updated) persona files on
    resume — storing it would freeze the persona's core text at
    first-chat time and silently resurrect deleted personas. Only the
    `entry_story` is kept: it has no other form on disk, and resuming
    must continue the same scene (see the module notes).
    """
    now = now or time.time()
    persist = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in _PERSIST_ROLES and isinstance(m.get("content"), str)
    ]
    path = _chat_path(persona_id, chat_id)
    with _LOCK:
        existing = _read_record(path) if path.is_file() else None
        existing_created = (existing or {}).get("created") or now
        record = {
            "id": chat_id,
            "persona": persona_id,
            "created": existing_created,
            "updated": now,
            "ended": ended,
            "entry_story": entry_story,
            "messages": persist,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: readers (list/load) never see a half-written file.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        _replace(tmp, path)


def _replace(tmp: Path, path: Path) -> None:
    """tmp.replace(path), retried briefly on a Windows sharing violation.

    Windows refuses to replace a file while another handle has it open —
    a list/load reading it, or the antivirus or search indexer scanning
    the file just written — and the save used to fail outright (seen
    when regenerating many times in a row). Those locks last
    milliseconds: a few short retries (about 0.6 s in all) ride them out.
    """
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * 2**attempt)


def delete_chat(persona_id: str, chat_id: str) -> bool:
    """Remove one saved chat. Returns True if a file was deleted."""
    path = _chat_path(persona_id, chat_id)
    with _LOCK:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False


def delete_persona_chats(persona_id: str) -> None:
    """Remove every saved chat of one persona (the persona was deleted)."""
    folder = _chat_path(persona_id, "x").parent  # validated id
    # rmtree is the one operation here that removes a whole folder, so
    # require it to be a direct child of CHATS_DIR after resolution
    # (on Windows "..." would otherwise resolve to CHATS_DIR itself).
    if folder.resolve().parent != CHATS_DIR.resolve():
        raise ValueError(f"unsafe id: {persona_id!r}")
    with _LOCK:
        shutil.rmtree(folder, ignore_errors=True)
