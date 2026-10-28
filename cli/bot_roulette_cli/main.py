"""``bot-roulette``: one command, one subcommand per job.

Everything the library leaves to its caller is decided here: the home
folder (BOT_ROULETTE_HOME, else the current directory) holds .env
(MODEL and OPENROUTER_API_KEY, or LLM_BASE_URL for your own
OpenAI-compatible server), config.yaml and personas/, and new
datasets go to its datasets/ folder unless --out says otherwise.
"""

import argparse
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError

from bot_roulette import (
    LLM,
    Config,
    Persona,
    PersonaIdentity,
    build_dataset,
    build_persona_from_cv,
    build_random_persona,
    calibrate_voice,
    evaluate_dataset,
    find_enriched_personas,
    live_paired_scenario,
    load_config,
    load_persona,
    rebuild_from_notes,
    resume_development,
)
from bot_roulette.chat import DEFAULT_MAX_TURNS

EXIT_COMMANDS = {"/quit", "/exit", "/q"}


class _CliConfig(Config):
    # The web app's section: the web app checks it, the CLI only skips it.
    web: dict | None = None


def home_dir() -> Path:
    return Path(os.environ.get("BOT_ROULETTE_HOME") or Path.cwd()).resolve()


def make_llm() -> LLM:
    """The model from the environment (.env included): on your own
    OpenAI-compatible server when LLM_BASE_URL is set (LLM_API_KEY only if
    it needs a key), else on OpenRouter. Exits with a pointer to .env when
    something is missing."""
    model = os.environ.get("MODEL", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    key_name = "LLM_API_KEY" if base_url else "OPENROUTER_API_KEY"
    key = os.environ.get(key_name, "").strip()
    required = (("MODEL", model),) if base_url else (("OPENROUTER_API_KEY", key), ("MODEL", model))
    missing = [name for name, value in required if not value]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} not set: add it to {home_dir() / '.env'}"
        )
    if not base_url:
        return LLM.openrouter(api_key=key, model=model)
    try:
        return LLM.openai_compatible(base_url, model=model, api_key=key)
    except ValueError as e:
        raise SystemExit(f"LLM_BASE_URL in {home_dir() / '.env'}: {e}")


def app_config(home: Path) -> Config:
    try:
        return load_config(home / "config.yaml", schema=_CliConfig)
    except RuntimeError as e:
        raise SystemExit(str(e))


# --- chat ------------------------------------------------------------------


def run_chat(args, home: Path) -> None:
    llm = make_llm()

    # PAIRED has no stored form: generate this session's entry story for
    # right now, exactly as the web app does.
    print("--- setting the scene… ---")
    scenario_text = live_paired_scenario(llm, load_persona(args.config).core)

    persona = Persona(
        llm,
        config_path=args.config,
        scenario_text=scenario_text,
        use_judge=not args.no_judge,
    )

    print(f"--- Chatting with {persona.name} (type /quit to exit) ---\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n--- Conversation ended ---")
            sys.exit(0)
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("--- Conversation ended ---")
            break
        answer = persona(user_input)
        if args.debug:
            print(f'{persona.name}: <state>{answer.state}</state>')
            print(f'{persona.name}: <thoughts>{answer.thoughts}</thoughts>')
            if answer.regenerations:
                print(
                    f'[judge: {answer.regenerations} draft(s) rejected'
                    + (
                        ''
                        if answer.verdict.passed
                        else f'; final still FAIL: {answer.verdict.violation}'
                    )
                    + ']'
                )
        print(f'{persona.name}: {answer.reply}')
        print()
        if answer.left:
            print(f"--- {persona.name} left the chat ---")
            break


# --- persona creation ------------------------------------------------------


def run_cv(args, home: Path) -> None:
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        args.parser.error(f"pdf file not found: {pdf_path}")
    config = app_config(home)
    build_persona_from_cv(
        make_llm(), pdf_path, root=home / "personas",
        creation=config.creation, max_turns=args.max_turns,
    )


def run_random(args, home: Path) -> None:
    try:
        identity = PersonaIdentity(
            first_name=args.first_name, surname=args.surname, sex=args.sex,
            country=args.country, chat_language=args.chat_language,
        )
    except ValidationError as e:
        args.parser.error("; ".join(
            f"--{str(err['loc'][0]).replace('_', '-')} "
            f"{err['msg'].removeprefix('Value error, ')}"
            for err in e.errors()
        ))
    config = app_config(home)
    build_random_persona(
        make_llm(), identity=identity, root=home / "personas",
        creation=config.creation, random_personas=config.random_personas,
        seed=args.seed, max_turns=args.max_turns,
    )


def run_enrich(args, home: Path) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        args.parser.error(f"config not found: {config_path}")
    creation = app_config(home).creation
    llm = make_llm()
    spec = load_persona(config_path)
    if args.rebuild:
        try:
            rebuild_from_notes(llm, config_path, creation=creation)
        except (ValueError, FileNotFoundError) as e:
            # A wrong target (a v0.1 root, missing notes or parent): say
            # why in one line, like the resume path below.
            raise SystemExit(str(e))
    elif not spec.parent or spec.parent == "none":
        try:
            resume_development(llm, config_path, creation=creation)
        except ValueError as e:
            raise SystemExit(str(e))
    else:
        ok = calibrate_voice(llm, config_path, creation=creation)
        if not ok:
            raise SystemExit(
                "voice calibration did not rewrite HOW YOU TEXT "
                "(no partners, chats too short, or empty rewrite)"
            )


# --- datasets --------------------------------------------------------------


def run_dataset(args, home: Path) -> None:
    available = find_enriched_personas(home / "personas")
    if args.personas:
        by_id = {c.parent.name: c for c in available}
        missing = [p for p in args.personas if p not in by_id]
        if missing:
            args.parser.error(
                f"unknown or un-enriched persona(s): {', '.join(missing)}. "
                f"Available: {', '.join(sorted(by_id)) or 'none'}"
            )
        configs = [by_id[p] for p in args.personas]
    else:
        configs = available
    if len(configs) < 2:
        args.parser.error(
            f"need at least 2 enriched personas, found {len(configs)}"
        )

    out_dir = (
        Path(args.out)
        if args.out
        else home / "datasets" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    try:
        build_dataset(
            make_llm(),
            configs,
            out_dir=out_dir,
            repeats=args.repeats,
            max_turns=args.turns,
            use_judge=not args.no_judge,
            rng=random.Random(args.seed),
        )
    except KeyboardInterrupt:
        # Conversations are flushed as they finish, so an interrupted run
        # still leaves a usable (smaller) dataset behind.
        logger.warning(f"interrupted; partial dataset kept in {out_dir}")
    except Exception:
        logger.error(traceback.format_exc())
        raise
    print(f"\nDataset: {out_dir}")
    print(f"Evaluate it with:  bot-roulette evaluate {out_dir}")


def _print_safely(text: str) -> None:
    """Print text that may contain block-bar and emoji characters.

    The report is written to disk as UTF-8, but stdout on Windows is
    often cp1252 (and always is when the output is piped), where the
    bars and ✅/⚠️ raise UnicodeEncodeError — losing the whole report
    after minutes of LLM calls. Fall back to the console's encoding with
    unmappable characters replaced.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def run_evaluate(args, home: Path) -> None:
    use_rubric = not args.no_rubric
    evaluate_dataset(
        args.dataset,
        llm=make_llm() if use_rubric else None,
        personas_dir=home / "personas",
        use_rubric=use_rubric,
    )
    dataset_dir = Path(args.dataset)
    print()
    _print_safely((dataset_dir / "report.md").read_text(encoding="utf-8"))
    print(f"Report saved: {dataset_dir / 'report.md'}")
    print(f"Scores saved: {dataset_dir / 'evaluation.json'}")


# --- the command -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bot-roulette",
        description="Chat with simulated people, create them, and measure them. "
        "Run from the folder holding .env, config.yaml and personas/ "
        "(or set BOT_ROULETTE_HOME).",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, func, **kwargs):
        sub = commands.add_parser(name, **kwargs)
        sub.set_defaults(func=func, parser=sub)
        return sub

    p = command("chat", run_chat, help="Chat with a persona.", description="Chat with a persona.")
    p.add_argument(
        "config",
        help="Path to the persona config YAML file (use an ENRICHED persona).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Also print the persona's <state> and <thoughts>.",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable the consistency judge (faster, cheaper).",
    )

    p = command(
        "cv", run_cv,
        help="Build a chat-ready persona from a CV.",
        description="Build a chat-ready persona from a CV: extraction, "
        "interview, enrichment.",
    )
    p.add_argument("--pdf", required=True, help="Path to the CV pdf file.")
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Interviewer question budget (default: "
        "creation.interview_questions in config.yaml).",
    )

    p = command(
        "random", run_random,
        help="Invent a random persona and develop it.",
        description="Invent a persona from scratch, around the name, sex and "
        "country you give and a random background, and develop it: "
        "generation, interview, enrichment, voice calibration.",
    )
    p.add_argument("--first-name", required=True, help="The persona's first name.")
    p.add_argument("--surname", required=True, help="The persona's surname.")
    p.add_argument("--sex", required=True, choices=("female", "male"))
    p.add_argument("--country", required=True, help="Where the persona lives, e.g. Italy.")
    p.add_argument(
        "--chat-language",
        default=None,
        help="A language the persona must be able to chat in, e.g. Italian "
        "(default: whatever is realistic for their life).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the background traits (default: random).",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Interviewer question budget (default: "
        "creation.interview_questions in config.yaml).",
    )

    p = command(
        "enrich", run_enrich,
        help="Re-run voice calibration, rebuild, or resume a persona.",
        description="Re-run voice calibration, rebuild an enriched core "
        "from saved field notes, or finish a v0.1 root whose creation run "
        "died or was stopped.",
    )
    p.add_argument(
        "config",
        help="Path to an enriched persona YAML, e.g. "
        "personas/francesca_20260906_221217/francesca.yaml — or to a v0.1 "
        "root YAML (e.g. personas/francesca/francesca.yaml) to resume its "
        "development, reusing its saved interview transcript.",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-run the enricher on the v0.1 parent + this folder's "
        "field notes; writes a new enriched folder. Default is voice "
        "calibration only (rewrites HOW YOU TEXT in place).",
    )

    p = command(
        "dataset", run_dataset,
        help="Generate conversations between personas.",
        description="Generate a dataset of conversations between personas.",
    )
    p.add_argument(
        "--personas",
        nargs="+",
        metavar="FOLDER",
        help="Persona folder ids to include (default: every enriched persona).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Conversations per pair (default: 1).",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max turns per conversation (default: {DEFAULT_MAX_TURNS}).",
    )
    p.add_argument(
        "--out",
        help="Output directory (default: datasets/<timestamp>).",
    )
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable the consistency judge (faster, and no regeneration "
             "drafts). The report then says the judge did not run rather "
             "than showing a clean sheet.",
    )
    p.add_argument(
        "--seed",
        type=int,
        # Only the conversation MOMENTS are random: pairs are every
        # unordered combination, and the opener alternates per repeat.
        help="Seed for the random conversation moments.",
    )

    p = command(
        "evaluate", run_evaluate,
        help="Evaluate a conversation dataset.",
        description="Evaluate the quality of a conversation dataset.",
    )
    p.add_argument("dataset", help="Dataset directory (see `bot-roulette dataset`).")
    p.add_argument(
        "--no-rubric",
        action="store_true",
        help="Skip the LLM rubric pass; compute mechanical metrics only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    home = home_dir()
    load_dotenv(home / ".env", override=True)
    # The library is quiet unless asked: its progress logs are this
    # tool's output.
    logger.enable("bot_roulette")
    args.func(args, home)


if __name__ == "__main__":
    main()
