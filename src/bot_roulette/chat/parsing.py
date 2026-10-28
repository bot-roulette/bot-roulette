"""Parsing the persona model's raw output.

The persona model answers in three tagged parts, <state>, <thoughts> and
<reply>, and marks chat events inside the reply: [NEXT] splits bubbles,
[LEAVE] closes the chat (see framework.py for the format it is asked
for). This module turns that raw text into what each consumer needs:
parse_response() for the three parts, split_bubbles() for what the
other person sees, and format_response() to rebuild a raw response
after an edit.
"""

import re


# Tag parsing. The model does not always emit perfectly formed
# <state>/<thoughts>/<reply> blocks: tags may be unclosed, duplicated,
# mismatched (`<thoughts>...</thoughts >`), capitalized, singular
# (`<thought>`), or missing entirely. The regexes below are forgiving:
# a block starts at its opening tag and runs until its own closing tag,
# the NEXT opening tag of any known kind, or the end of the text.
_TAG_NAMES = {
    "state": r"state",
    "thoughts": r"thoughts?",   # tolerate <thought>
    "reply": r"repl(?:y|ies)",  # tolerate <replies>
}
# Attribute part of a tag, shared by every tag regex below. It must be
# real grammar (`name="value"` pairs), never a free-for-all: an earlier
# `(?:\s[^<>]*)?` matched ANY bracket-free run, so ordinary prose with
# two comparisons ("a < b ma 5 > 3") or a comparison whose operands
# happen to be tag words ("il mio stato < state oggi > male") looked
# like a tag and was silently deleted. Requiring `=value` keeps prose
# intact while tags WITH attributes still parse.
_TAG_ATTRS = (
    r"(?:\s+[A-Za-z_:][\w:.-]*\s*=\s*"
    r"(?:\"[^\"<>]*\"|'[^'<>]*'|[^\s<>\"']+))*"
)
_ANY_OPEN = rf"<\s*(?:state|thoughts?|repl(?:y|ies)){_TAG_ATTRS}\s*>"


def _tolerant_re(name_pattern: str) -> re.Pattern:
    return re.compile(
        rf"<\s*{name_pattern}{_TAG_ATTRS}\s*>"  # opening tag (attrs ok)
        rf"(.*?)"                            # content, lazy
        rf"(?:<\s*/\s*{name_pattern}\s*>"    # its own closing tag...
        rf"|(?={_ANY_OPEN})"                 # ...or next opening tag...
        rf"|\Z)",                            # ...or end of text
        re.DOTALL | re.IGNORECASE,
    )


STATE_RE = _tolerant_re(_TAG_NAMES["state"])
THOUGHTS_RE = _tolerant_re(_TAG_NAMES["thoughts"])
REPLY_RE = _tolerant_re(_TAG_NAMES["reply"])
# Any stray/duplicated/orphaned tag left inside extracted text, e.g.
# "<reply><reply>...", a dangling "</state>", or "<thought>". Also
# catches tags the model INVENTS (<message>, <bubble>, <response>,
# <br>): those used to survive into the bubbles, which is why replies
# sometimes started with "<". Deliberately narrow so real text is never
# eaten: the tag name must be a bare word (letters/digits/_/-), so "<3",
# "a < b" and "<--" are left alone.
#
# The attribute part is the shared _TAG_ATTRS grammar (see above): only
# whitespace + `name="value"` pairs, then the closing bracket. Prose
# containing comparisons ("a < b ma 5 > 3") is not a tag.
STRAY_TAGS_RE = re.compile(
    rf"<\s*/?\s*[A-Za-z][\w-]*{_TAG_ATTRS}\s*/?\s*>"
)
# ...except the scenario markers, which are meaningful and are handled
# later (NEXT splits bubbles, LEAVE ends the chat). The model writes them
# as <NEXT> often enough — by analogy with the real tags — that stripping
# them here would silently merge the bubbles instead of splitting them.
_KEEP_MARKERS_RE = re.compile(r"<\s*/?\s*(?:next|leave)\s*/?\s*>", re.IGNORECASE)
# Markdown code fences the model sometimes wraps the whole output in.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|```\s*$")


def _clean(text: str) -> str:
    """Remove stray tags left inside extracted text, keeping markers.

    Removed tags become a space, not nothing: "<bubble>ciao</bubble>
    <bubble>come va</bubble>" would otherwise collapse into "ciaocome
    va". Runs of whitespace are then squeezed back down.
    """
    out = STRAY_TAGS_RE.sub(
        lambda m: m.group(0) if _KEEP_MARKERS_RE.fullmatch(m.group(0)) else " ",
        text,
    )
    # Keep line structure (it separates bubbles) but drop the extra
    # spaces the substitution above may have introduced.
    out = re.sub(r"[^\S\n]+", " ", out)
    return re.sub(r" *\n *", "\n", out).strip()


def parse_response(response: str) -> tuple[str, str, str]:
    """Split a raw model response into (state, thoughts, reply).

    Robust to malformed output. Strategy:
    1. Strip markdown fences.
    2. Extract each block with tolerant regexes (first match wins;
       content of duplicated blocks is joined).
    3. The reply is the <reply> block if present; otherwise everything
       left after removing the state/thoughts blocks. If neither exists
       the reply is EMPTY: the thoughts are private and are never
       promoted to a message (Persona.__call__ retries instead).
    """
    text = _FENCE_RE.sub("", response.strip()).strip()

    def extract(pattern: re.Pattern) -> str:
        parts = [m.group(1) for m in pattern.finditer(text)]
        return _clean("\n".join(p for p in parts if p.strip()))

    state = extract(STATE_RE)
    thoughts = extract(THOUGHTS_RE)
    reply = extract(REPLY_RE)

    if not reply:
        # No <reply> block at all: whatever is outside the other blocks
        # is the reply (handles "state/thoughts then bare text").
        leftover = THOUGHTS_RE.sub("", STATE_RE.sub("", text))
        reply = _clean(leftover)
    # Still empty means the model emitted only state/thoughts. Leave it
    # empty: falling back to the thoughts would send the inner monologue
    # — which the other person must never see — as a chat message.

    return state, thoughts, reply


def format_response(state: str, thoughts: str, reply: str) -> str:
    """Reassemble a raw model response string from its parsed parts.

    Inverse of parse_response: used by Persona.set_last_turn to rebuild
    the assistant message stored in history after a spectator edit.
    """
    return (
        f"<state>{state}</state>\n"
        f"<thoughts>{thoughts}</thoughts>\n"
        f"<reply>{reply}</reply>"
    )

# Scenario-level markers ([NEXT] bubble separator, [LEAVE] chat close).
# The model doesn't always emit them literally: tolerate case/spacing
# variants ([next], [ LEAVE ]), doubled/other brackets ([[NEXT]],
# (LEAVE), {next}, <NEXT>, </NEXT>, <NEXT/>), and — for the unambiguous
# all-caps form — a missing bracket on one side ("[NEXT", "LEAVE]").
# A bare unbracketed word is never matched: "next" and "leave" are
# common English words.
#
# Angle brackets matter: the prompt is full of <state>/<thoughts>/<reply>
# tags, so the model quite often writes <NEXT> by analogy. Without them
# the marker stayed in the text and the bubbles never split.
_OPEN = r"[\[({<]"
_CLOSE = r"[\])}>]"


def _marker_re(word: str) -> re.Pattern:
    return re.compile(
        # bracketed, any case/brackets; optional inner slashes tolerate
        # tag-style spellings like </NEXT> and <NEXT/>
        rf"(?i:{_OPEN}+\s*/?\s*{word}\s*/?\s*{_CLOSE}+)"
        rf"|{_OPEN}+\s*{word.upper()}\b"       # opening bracket only
        rf"|\b{word.upper()}\s*{_CLOSE}+"      # closing bracket only
    )


NEXT_RE = _marker_re("next")
LEAVE_RE = _marker_re("leave")


def split_bubbles(reply: str) -> list[str]:
    """The bubbles a parsed reply shows: [LEAVE] removed, one bubble per
    [NEXT]-separated chunk AND per line, blank ones dropped.

    The single definition of what the other person sees: live turns and
    replayed chats split the same way.
    """
    reply = LEAVE_RE.sub("", reply)
    return [
        line.strip()
        for chunk in NEXT_RE.split(reply)
        for line in chunk.split("\n")
        if line.strip()
    ]
