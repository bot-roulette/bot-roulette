"""The web app end to end, in order: first unconfigured, then the
settings panel configures it, then a chat runs against a fake model.
Everything happens inside a temporary home folder."""

import base64
import shutil

import pytest
from fastapi import WebSocketDisconnect


def reply(text):
    return f"<state>calm</state>\n<thoughts>mah</thoughts>\n<reply>{text}</reply>"


def basic(password, username=""):
    """The Authorization header a browser sends once it has asked for the code."""
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def test_a_session_must_open_with_a_start_message(web):
    with web.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "persona": "monica_20260907_060117"})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_an_access_code_guards_every_page_and_the_chat(web, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web.server, "ACCESS_CODE", "caffè")
    client = TestClient(web.server.app)
    for headers in ({}, basic("wrong"), {"Authorization": "Basic %%%"}, {"Authorization": "Bearer x"}):
        for path in ("/", "/personas", "/settings"):
            r = client.get(path, headers=headers)
            assert r.status_code == 401
            assert r.headers["www-authenticate"].startswith('Basic realm="bot-roulette"')
    # The username is ignored; the code may be any UTF-8 text.
    assert 'id="settings-panel"' in client.get("/", headers=basic("caffè", username="me")).text
    assert client.get("/personas", headers=basic("caffè")).status_code == 200
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "ended", "reason": "access code required — reload the page"}
    with client.websocket_connect("/ws", headers=basic("caffè")) as ws:
        ws.send_json({"type": "start", "persona": "nobody"})
        assert ws.receive_json() == {"type": "ended", "reason": "unknown persona"}


def test_unconfigured_state_is_reported(web):
    body = web.client.get("/settings").json()
    assert body == {"server": "", "model": "7440", "api_key_hint": "", "openrouter_key_hint": "",
                    "configured": False, "model_known": False}
    personas = web.client.get("/personas").json()
    assert personas["configured"] is False
    assert {p["id"] for p in personas["personas"]} == {
        "francesca_20260907_064108", "monica_20260907_060117", "simona_20260906_225636",
    }


def test_generation_options_are_the_background_lists(web):
    response = web.client.get("/personas/generation-options")
    assert response.status_code == 200
    body = response.json()
    assert body["age_bands"] == web.server.CONFIG.random_personas.age_bands
    # Who the person is is typed in, never offered from a list.
    assert not {"sexes", "countries", "names", "surnames"} & set(body)


IDENTITY = {"first_name": "Anna", "surname": "Rossi", "sex": "female", "country": "Italy"}


def test_generation_request_forwards_identity_and_fixed_traits_to_the_job(web, monkeypatch):
    monkeypatch.setattr(web.server, "is_configured", lambda: True)
    captured = {}

    def start_job(kind, run):
        captured.update(kind=kind, run=run)
        return "test-job"

    monkeypatch.setattr(web.server, "_start_job", start_job)
    monkeypatch.setattr(
        web.server, "build_random_persona", lambda llm, **kwargs: captured.update(kwargs)
    )
    fixed = {
        "age_band": web.server.CONFIG.random_personas.age_bands[0],
        "settlement": "  a fishing village\nnear Bari ",
    }
    identity = {**IDENTITY, "first_name": " Hiroshi ", "chat_language": "Italian"}
    response = web.client.post(
        "/personas/generate", json={"identity": identity, "fixed_traits": fixed}
    )
    assert response.status_code == 200 and response.json() == {"job": "test-job"}
    captured["run"]()
    assert captured["kind"] == "generate"
    # Listed values arrive as they are, your own words tidied.
    assert captured["fixed_traits"] == {**fixed, "settlement": "a fishing village near Bari"}
    assert (captured["identity"].first_name, captured["identity"].chat_language) == ("Hiroshi", "Italian")


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({}, "identity: Field required"),
        ({"identity": {**IDENTITY, "surname": "  "}}, "surname: can't be empty"),
        ({"identity": {**IDENTITY, "sex": "other"}}, "identity.sex"),
        ({"identity": IDENTITY, "fixed_traits": {"country": "Italy"}}, "unknown fixed traits: country"),
        ({"identity": IDENTITY, "fixed_traits": {"age_band": "1-2"}}, "invalid age_band"),
        ({"identity": IDENTITY, "fixed_traits": {"age_band": "16"}}, "invalid age_band"),
        ({"identity": IDENTITY, "fixed_traits": {"work_area": "x" * 151}}, "at most 150"),
    ],
)
def test_generation_requests_are_checked_before_a_job_starts(web, monkeypatch, payload, needle):
    monkeypatch.setattr(web.server, "is_configured", lambda: True)
    monkeypatch.setattr(web.server, "_start_job", lambda *args: pytest.fail("a job started"))
    response = web.client.post("/personas/generate", json=payload)
    assert response.status_code == 400
    assert needle in response.json()["detail"]


def test_unconfigured_server_refuses_model_work(web):
    r = web.client.post("/personas/generate")
    assert r.status_code == 503 and "settings" in r.json()["detail"]
    with web.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "persona": "monica_20260907_060117"})
        assert ws.receive_json() == {"type": "ended", "reason": web.server.NOT_CONFIGURED}
    with web.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "persona": "monica_20260907_060117",
                      "persona_b": "simona_20260906_225636"})
        assert ws.receive_json() == {"type": "ended", "reason": web.server.NOT_CONFIGURED}


@pytest.mark.parametrize(
    "payload, status, needle",
    [
        ({}, 400, "nothing to change"),
        ({"model": "7440"}, 400, "nothing to change"),
        ({"model": "nope/model"}, 400, "no model called"),
        ({"api_key": "sk-or-v1-wrong"}, 400, "rejected"),
        ({"api_key": "sk-or-v1-down"}, 502, "nothing was saved"),
        ({"api_key": "sk-or-v1 with space"}, 400, "doesn't look like"),
        ({"api_key": "sk-or-v1-\u00e8"}, 400, "doesn't look like"),
        ({"api_key": "x" * 300}, 422, ""),
        ({"model": "z-ai/glm-5.2", "api_key": "sk-or-v1-wrong"}, 400, "rejected"),
        # Another server: a well-formed address, and a model it has.
        ({"server": "localhost:11434/v1", "model": "gemma3"}, 400, "not an http"),
        ({"server": "http://localhost:11434/v1?x=1", "model": "gemma3"}, 400, "not an http"),
        ({"server": "http://[::1/v1", "model": "gemma3"}, 400, "not an http"),
        ({"server": "http://localhost:11434/v1", "model": "llama3"}, 400, "no model called 'llama3'"),
        ({"server": "http://localhost:11434/v1"}, 400, "no model called '7440'"),
        ({"server": "x" * 301}, 422, ""),
    ],
)
def test_rejected_settings_change_nothing(web, payload, status, needle):
    r = web.client.post("/settings", json=payload)
    assert r.status_code == status and needle in r.text
    assert not (web.home / ".env").exists()
    assert web.client.get("/settings").json()["configured"] is False


def test_saving_settings_writes_env_and_switches_the_live_llm(web):
    r = web.client.post("/settings", json={"model": " openrouter/auto:nitro ", "api_key": web.good_key})
    assert r.status_code == 200
    assert r.json() == {"server": "", "model": "openrouter/auto:nitro", "api_key_hint": "\u2026abcd",
                        "openrouter_key_hint": "\u2026abcd", "configured": True, "model_known": True}
    assert web.good_key not in r.text
    env = (web.home / ".env").read_text(encoding="utf-8")
    assert "MODEL=openrouter/auto:nitro\n" in env and f"OPENROUTER_API_KEY={web.good_key}\n" in env
    llm = web.server.llm
    llm.client._refresh_api_key()
    assert (llm.client.api_key, llm.resolve_model()) == (web.good_key, "openrouter/auto:nitro")

    r = web.client.post("/settings", json={"api_key": web.good_key_2})
    assert r.json()["api_key_hint"] == "\u2026wxyz"
    llm.client._refresh_api_key()
    assert llm.client.api_key == web.good_key_2


def test_model_suggestions_are_cached(web):
    before = web.openrouter.calls["models"]
    r = web.client.get("/settings/models")
    assert [m["id"] for m in r.json()["models"]] == [m["id"] for m in web.catalog]
    web.client.get("/settings/models")
    assert web.openrouter.calls["models"] - before <= 1


def test_a_chat_runs_on_the_configured_llm_and_is_saved_in_home(web, monkeypatch):
    fake = web.FakeClient("- an entry story", reply("ciao! [NEXT] chi sei?"), '{"passed": true}')
    monkeypatch.setattr(web.server.llm, "client", fake)
    with web.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "persona": "monica_20260907_060117",
                      "local_time": "2026-10-25T20:00"})
        while (msg := ws.receive_json())["type"] != "chat":
            pass
        chat_id = msg["id"]
        ws.send_json({"type": "message", "text": "ciao"})
        bubbles = []
        while (msg := ws.receive_json())["type"] != "done":
            if msg["type"] == "bubble":
                bubbles.append(msg["text"])
    assert bubbles == ["ciao!", "chi sei?"]
    assert {r["model"] for r in fake.requests} == {"openrouter/auto:nitro"}
    assert (web.home / "chats" / "monica_20260907_060117" / f"{chat_id}.json").exists()


def _persona(home, folder, name, **flags):
    """A minimal enriched persona folder in the web app's home."""
    d = home / "personas" / folder
    d.mkdir(parents=True)
    (d / "p.md").write_text(f"=== WHO YOU ARE ===\n{name}, 30.", encoding="utf-8")
    (d / "p.yaml").write_text(
        f"name: {name}\nfull_name: {name} Rossi\ncore: p.md\nparent: {name.lower()}\n"
        + "".join(f"{key}: {str(value).lower()}\n" for key, value in flags.items()),
        encoding="utf-8",
    )


def test_a_persona_with_no_one_to_calibrate_against_can_chat(web, monkeypatch):
    # The first persona skipped voice calibration (no one to text) and is
    # chat-ready anyway; one whose calibration didn't finish still waits.
    _persona(web.home, "lone_1", "Lone", voice_calibrated=False, chat_ready=True)
    _persona(web.home, "wait_1", "Wait", voice_calibrated=False, chat_ready=False)
    try:
        listed = {p["id"]: p for p in web.client.get("/personas").json()["personas"]}
        assert (listed["lone_1"]["chat_ready"], listed["lone_1"]["voice_calibrated"]) == (True, False)
        assert listed["wait_1"]["chat_ready"] is False
        with web.client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "persona": "wait_1"})
            assert ws.receive_json() == {
                "type": "ended", "reason": "persona is not ready yet (voice calibration pending)",
            }
        # Configured or not depends on the tests before; not the point here.
        monkeypatch.setattr(web.server, "is_configured", lambda: True)
        monkeypatch.setattr(web.server.llm, "model", "test-model")
        monkeypatch.setattr(web.server.llm, "client", web.FakeClient("- an entry story"))
        with web.client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "start", "persona": "lone_1", "local_time": "2026-10-25T20:00"})
            while (msg := ws.receive_json())["type"] != "chat":
                assert msg["type"] != "ended", msg
    finally:
        for folder in ("lone_1", "wait_1"):
            shutil.rmtree(web.home / "personas" / folder)


def test_a_spectator_edit_is_handed_on_without_leave_or_markers(web, monkeypatch):
    fake = web.FakeClient(
        "- a story", "- a story",
        reply("ciao"), '{"passed": true}',
        reply("eh?"), '{"passed": true}',
    )
    monkeypatch.setattr(web.server.llm, "client", fake)
    with web.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "persona": "monica_20260907_060117",
                      "persona_b": "simona_20260906_225636", "local_time": "2026-10-25T20:00"})
        while ws.receive_json()["type"] != "editable":
            pass
        # Leaving is the persona's own move: an edit can't make it leave.
        ws.send_json({"type": "edit", "who": "a", "reply": "ciao [NEXT] ci sei? [LEAVE]"})
        while (msg := ws.receive_json())["type"] != "editable":
            assert msg["type"] != "ended"
        ws.send_json({"type": "stop"})
    assert fake.requests[4]["messages"][-1]["content"] == "ciao\nci sei?"


@pytest.mark.parametrize("hours, trusted", [(-36, True), (36, True), (-60, False), (60, False)])
def test_the_client_clock_is_trusted_within_two_days_either_way(web, hours, trusted):
    from datetime import datetime, timedelta

    when = datetime.now().replace(microsecond=0) + timedelta(hours=hours)
    moment = web.server._session_moment({"local_time": when.isoformat()})
    assert (moment == when) is trusted


def test_openrouter_down(web):
    web.openrouter.models_down = True
    web.openrouter_module._catalog = None
    try:
        assert web.client.get("/settings/models").status_code == 502
        assert web.client.get("/settings").json()["model_known"] is None
    finally:
        web.openrouter.models_down = False


def test_the_page_ships_with_the_package(web):
    page = web.client.get("/")
    assert page.status_code == 200 and 'id="settings-panel"' in page.text
