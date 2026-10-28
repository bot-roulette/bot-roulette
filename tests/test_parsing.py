import pytest

from bot_roulette.chat.parsing import (
    LEAVE_RE,
    STRAY_TAGS_RE,
    format_response,
    parse_response,
    split_bubbles,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "<state>calm</state>\n<thoughts>ok then</thoughts>\n<reply>hi [NEXT] there</reply>",
            ("calm", "ok then", "hi [NEXT] there"),
        ),
        ("<state>s</state><thoughts>t</thoughts><reply>ciao", ("s", "t", "ciao")),
        ('<STATE>s</STATE><thought>t</thought><reply lang="it">ciao</reply>', ("s", "t", "ciao")),
        ("```xml\n<state>s</state><thoughts>t</thoughts><reply>ciao</reply>\n```", ("s", "t", "ciao")),
        ("<state>s</state><thoughts>t</thoughts>\nciao", ("s", "t", "ciao")),
        # Thoughts are never promoted to a reply.
        ("<state>s</state><thoughts>secret</thoughts>", ("s", "secret", "")),
        ("<state>s</state><thoughts>t</thoughts><reply>a</reply><reply>b</reply>", ("s", "t", "a\nb")),
        (
            "<state>s</state><thoughts>t</thoughts><reply><bubble>ciao</bubble><bubble>come va</bubble></reply>",
            ("s", "t", "ciao come va"),
        ),
        # Prose with comparisons, hearts and markers survives.
        ("<state>s</state><thoughts>t</thoughts><reply>a < b ma 5 > 3</reply>", ("s", "t", "a < b ma 5 > 3")),
        (
            "<state>s</state><thoughts>t</thoughts><reply>ti voglio bene <3 [NEXT] <NEXT> ok</reply>",
            ("s", "t", "ti voglio bene <3 [NEXT] <NEXT> ok"),
        ),
    ],
)
def test_parse_response(raw, expected):
    assert parse_response(raw) == expected


@pytest.mark.parametrize(
    "text, bubbles",
    [
        ("hi [NEXT] there", ["hi", "there"]),
        ("a\nb\n\nc", ["a", "b", "c"]),
        ("uno <NEXT> due </NEXT> tre", ["uno", "due", "tre"]),
        ("bye [LEAVE]", ["bye"]),
        ("[next]x[ NEXT ]y", ["x", "y"]),
        ("NEXT] a", ["a"]),
        ("next time", ["next time"]),
    ],
)
def test_split_bubbles(text, bubbles):
    assert split_bubbles(text) == bubbles


@pytest.mark.parametrize(
    "text, leaves",
    [("[LEAVE]", True), ("(leave)", True), ("LEAVE]", True), ("{Leave}", True),
     ("leave me alone", False), ("[NEXT]", False)],
)
def test_leave_marker(text, leaves):
    assert bool(LEAVE_RE.search(text)) is leaves


def test_format_response_round_trips():
    assert parse_response(format_response("st", "th", "uno [NEXT] due")) == (
        "st", "th", "uno [NEXT] due",
    )


@pytest.mark.parametrize("prose", ["a < b ma 5 > 3", "ti voglio bene <3", "<--"])
def test_stray_tag_detector_ignores_prose(prose):
    assert not STRAY_TAGS_RE.search(prose)
