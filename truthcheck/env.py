"""Load settings from a local `.env` file so the API key never has to be typed or committed.

The file lives in the project root (next to app.py), is listed in .gitignore, and holds lines like
TRUTHCHECK_API_KEY=... Real environment variables always win over the file.
Set TRUTHCHECK_NO_DOTENV=1 to skip loading (the test-suite does this).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(path: Optional[Path] = None) -> int:
    """Read KEY=VALUE lines into os.environ without overriding existing variables. Returns how many were set."""
    if os.environ.get("TRUTHCHECK_NO_DOTENV"):
        return 0
    try:
        text = Path(path or DEFAULT_PATH).read_text(encoding="utf-8-sig")
    except OSError:
        return 0
    n = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and v and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n
