import pytest

from bot_roulette import LLM
from bot_roulette_cli import main as cli

COMMANDS = ("chat", "cv", "random", "enrich", "dataset", "evaluate")
RANDOM = ["random", "--first-name", "Anna", "--surname", "Rossi", "--sex", "female", "--country", "Italy"]


def reply(text):
    return f"<state>calm</state>\n<thoughts>mah</thoughts>\n<reply>{text}</reply>"


def run(argv, capsys):
    """Run the command; returns (exit code or SystemExit message, stdout, stderr)."""
    try:
        cli.main(argv)
        code = 0
    except SystemExit as e:
        code = e.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def fake_llm(home, monkeypatch):
    def install(*replies):
        fake = home.FakeClient(*replies)
        monkeypatch.setattr(cli, "make_llm", lambda: LLM(fake, "test-model"))
        return fake

    return install


def test_help_lists_every_command(capsys):
    code, out, _ = run(["--help"], capsys)
    assert code == 0
    for command in COMMANDS:
        assert command in out


@pytest.mark.parametrize("command", COMMANDS)
def test_help_needs_no_key_and_no_config(command, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BOT_ROULETTE_HOME", str(tmp_path))  # empty folder
    code, out, _ = run([command, "--help"], capsys)
    assert code == 0 and "usage: bot-roulette" in out


def test_a_missing_key_points_at_the_home_env(home, capsys):
    code, _, _ = run(["chat", str(home.anna)], capsys)
    assert "OPENROUTER_API_KEY and MODEL not set" in code
    assert str(home.path / ".env") in code


def test_chat(home, fake_llm, monkeypatch, capsys):
    fake = fake_llm("- an entry story", reply("uno [NEXT] due"))
    answers = iter(["ciao", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    code, out, _ = run(["chat", str(home.anna), "--no-judge"], capsys)
    assert code == 0
    assert "--- Chatting with Anna" in out
    assert "Anna: uno\ndue" in out
    assert out.rstrip().endswith("--- Conversation ended ---")
    assert len(fake.requests) == 2


def test_dataset_then_evaluate(home, fake_llm, capsys):
    fake_llm("- story A", "- story B", reply("ciao"))
    code, out, _ = run(["dataset", "--turns", "1", "--no-judge", "--seed", "1"], capsys)
    assert code == 0
    (run_dir,) = (home.path / "datasets").iterdir()
    assert f"bot-roulette evaluate {run_dir}" in out

    code, out, _ = run(["evaluate", str(run_dir), "--no-rubric"], capsys)
    assert code == 0 and "Report saved" in out
    assert (run_dir / "report.md").exists()


def test_a_dataset_needs_two_personas(home, capsys):
    code, _, err = run(["dataset", "--personas", "anna_1"], capsys)
    assert code == 2 and "need at least 2 enriched personas" in err


def test_a_developed_root_is_not_resumed(home, fake_llm, capsys):
    fake = fake_llm()
    code, _, _ = run(["enrich", str(home.anna_root)], capsys)
    assert "already developed" in code
    assert fake.requests == []


def test_config_problems_are_reported_by_name(home, fake_llm, capsys):
    config = home.path / "config.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("interview_questions:", "intervew_questions:"),
        encoding="utf-8",
    )
    code, _, _ = run(RANDOM, capsys)
    assert "intervew_questions" in code


def test_random_builds_the_persona_it_is_given(home, fake_llm, monkeypatch, capsys):
    fake_llm()
    captured = {}
    monkeypatch.setattr(cli, "build_random_persona", lambda llm, **kwargs: captured.update(kwargs))
    code, _, _ = run(RANDOM + ["--chat-language", "English", "--seed", "3"], capsys)
    assert code == 0
    identity = captured["identity"]
    assert (identity.first_name, identity.sex, identity.chat_language) == ("Anna", "female", "English")
    assert captured["seed"] == 3


@pytest.mark.parametrize(
    "argv, problem",
    [
        (RANDOM[:5], "--sex"),  # who the persona is can't be left out
        (RANDOM + ["--first-name", "  "], "--first-name can't be empty"),
    ],
)
def test_random_needs_a_usable_identity(home, argv, problem, capsys):
    code, _, err = run(argv, capsys)
    assert code == 2 and problem in err


def test_the_home_env_wins_over_the_os_environment(home, monkeypatch, capsys):
    (home.path / ".env").write_text("OPENROUTER_API_KEY=sk-from-env\nMODEL=from/env\n", encoding="utf-8")
    monkeypatch.setenv("MODEL", "7440")  # a stray system-wide variable
    code, _, _ = run(["dataset", "--personas", "nobody"], capsys)  # loads .env, then stops
    assert code == 2
    assert cli.make_llm().resolve_model() == "from/env"


def test_a_local_server_needs_only_a_model(home, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-os")
    (home.path / ".env").write_text("LLM_BASE_URL=http://localhost:1234/v1\n", encoding="utf-8")
    code, _, _ = run(["chat", str(home.anna)], capsys)
    assert code.startswith("MODEL not set") and str(home.path / ".env") in code

    monkeypatch.setenv("MODEL", "qwen2.5-vl-7b-instruct")
    llm = cli.make_llm()
    llm.client._refresh_api_key()
    assert str(llm.client.base_url) == "http://localhost:1234/v1/"
    assert llm.resolve_model() == "qwen2.5-vl-7b-instruct"
    assert llm.client.auth_headers == {"Authorization": "Bearer no-key"}  # never the system-wide key

    monkeypatch.setenv("LLM_BASE_URL", "localhost:1234/v1")
    with pytest.raises(SystemExit, match="LLM_BASE_URL"):
        cli.make_llm()
