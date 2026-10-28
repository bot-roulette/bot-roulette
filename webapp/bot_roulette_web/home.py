"""Where the web app's files live: one home folder holding .env,
config.yaml, personas/ and chats/.

The home is the BOT_ROULETTE_HOME environment variable if it is set,
else the current directory (the repository root, when run from a
checkout). It is resolved once, at import, to an absolute path:
everything the app reads or writes is under it, wherever the process
later runs from.
"""

import os
from pathlib import Path

HOME = Path(os.environ.get("BOT_ROULETTE_HOME") or Path.cwd()).resolve()
ENV_PATH = HOME / ".env"
CONFIG_PATH = HOME / "config.yaml"
PERSONAS_DIR = HOME / "personas"
CHATS_DIR = HOME / "chats"
