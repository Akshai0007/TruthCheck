import os

os.environ["TRUTHCHECK_NO_DOTENV"] = "1"   # tests must never read the developer's real .env

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("TRUTHCHECK_PROVIDER", "TRUTHCHECK_MODEL", "TRUTHCHECK_API_KEY", "TRUTHCHECK_BASE_URL",
              "GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
