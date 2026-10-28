"""bot-roulette-web: serve the web app with uvicorn.

Run it from the home folder (the one holding .env, config.yaml and
personas/), or set BOT_ROULETTE_HOME; see home.py.
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bot-roulette-web",
        description="Serve the bot-roulette web app. Run it from the folder "
        "holding .env, config.yaml and personas/, or set BOT_ROULETTE_HOME.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Interface to listen on (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument("--reload", action="store_true", help="Restart when the code changes (development).")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "bot_roulette_web.server:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
