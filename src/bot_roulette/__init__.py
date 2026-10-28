"""bot-roulette: simulated people you can text with.

Build a persona from a CV or from scratch, develop it through an
interview, and chat with it. Every model call goes through an LLM you
create and pass in; the library reads no environment variables and no
files of its own:

    from bot_roulette import LLM, Persona, live_paired_scenario, load_persona

    llm = LLM.openrouter(api_key="sk-or-...", model="z-ai/glm-5.2")
    # or any OpenAI-compatible server, e.g. a local one:
    # llm = LLM.openai_compatible("http://localhost:11434/v1", "qwen2.5vl:7b")
    config = "personas/monica_20260907_060117/monica.yaml"
    scene = live_paired_scenario(llm, load_persona(config).core)
    monica = Persona(llm, config, scenario_text=scene)
    print(monica("ciao!").reply)

Logging is off by default: call ``logger.enable("bot_roulette")``
(from loguru) to see progress logs.

The whole public API is importable from here. Underneath, the library
is split into areas, each building only on the ones listed before it:

    llm/         the model: LLM, complete(), OpenRouter checks
    config.py    the settings persona creation needs, and their loader
    persona/     a persona folder (load_persona...) and its scenarios
    chat/        talking to a persona: Persona, judge, parsing, conversations
    creation/    new personas: from a CV or from scratch, interview, enrichment
    evaluation/  conversation datasets and their evaluation
"""

from loguru import logger

from .chat import (
    DEFAULT_MAX_TURNS,
    LEAVE_RE,
    Conversation,
    Judge,
    Persona,
    Response,
    Turn,
    Verdict,
    format_response,
    parse_response,
    run_conversation,
    split_bubbles,
)
from .config import Config, CreationConfig, RandomPersonasConfig, load_config
from .creation import (
    PersonaIdentity,
    build_persona_from_cv,
    build_random_persona,
    calibrate_voice,
    develop_persona,
    enrich_persona,
    rebuild_from_notes,
    resume_development,
    validate_fixed_seed_traits,
)
from .evaluation import build_dataset, evaluate_dataset, load_dataset
from .llm import (
    LLM,
    Cancelled,
    OpenRouterUnavailable,
    ServerUnavailable,
    UnusableOutput,
    check_api_key,
    check_model,
    check_server_model,
    complete,
    extract_json,
    is_known_model,
    is_served_model,
    model_catalog,
    server_models,
    set_cancel_event,
)
from .persona import (
    PersonaSpec,
    find_chat_ready_personas,
    find_enriched_personas,
    generate_entry_story,
    live_paired_scenario,
    load_persona,
    paired_scenario,
)

__version__ = "0.1.0"

logger.disable(__name__)

__all__ = [
    "DEFAULT_MAX_TURNS",
    "LEAVE_RE",
    "LLM",
    "Cancelled",
    "Config",
    "Conversation",
    "CreationConfig",
    "Judge",
    "OpenRouterUnavailable",
    "Persona",
    "PersonaIdentity",
    "PersonaSpec",
    "RandomPersonasConfig",
    "Response",
    "ServerUnavailable",
    "Turn",
    "UnusableOutput",
    "Verdict",
    "build_dataset",
    "build_persona_from_cv",
    "build_random_persona",
    "calibrate_voice",
    "check_api_key",
    "check_model",
    "check_server_model",
    "complete",
    "develop_persona",
    "enrich_persona",
    "evaluate_dataset",
    "extract_json",
    "find_chat_ready_personas",
    "find_enriched_personas",
    "format_response",
    "generate_entry_story",
    "is_known_model",
    "is_served_model",
    "live_paired_scenario",
    "load_config",
    "load_dataset",
    "load_persona",
    "model_catalog",
    "paired_scenario",
    "parse_response",
    "rebuild_from_notes",
    "resume_development",
    "run_conversation",
    "server_models",
    "set_cancel_event",
    "split_bubbles",
    "validate_fixed_seed_traits",
]
