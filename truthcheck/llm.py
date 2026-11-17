"""Tiny LLM layer. Any object with .complete(system, user, temperature) works.

Supported out of the box:
  anthropic            -> Anthropic Messages API
  openai / groq / gemini / ollama / openrouter -> any OpenAI-compatible endpoint
  demo                 -> scripted offline stand-in (sample input only)

Free options: Groq and Gemini have free tiers, Ollama runs models locally.
Check each provider's docs for current model names and limits.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Protocol

import requests

OPENAI_COMPAT_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "ollama": "llama3.1"}


class LLM(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0,
                 max_tokens: int = 1024) -> str: ...


class _HTTPClient:
    def __init__(self, model: str, api_key: str, timeout: int = 60, retries: int = 3):
        self.model, self.api_key, self.timeout, self.retries = model, api_key, timeout, retries

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        last = "no attempt made"
        for attempt in range(self.retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = str(e)
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 529):  # busy / rate-limited: wait, then retry
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                wait = r.headers.get("Retry-After", "")
                time.sleep(min(float(wait), 30) if wait.replace(".", "", 1).isdigit() else 2 ** attempt)
                continue
            if r.status_code >= 400:  # wrong key, wrong model name...: retrying cannot help
                raise RuntimeError(f"LLM request rejected (HTTP {r.status_code}): {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"LLM request failed after {self.retries} tries: {last}")


class AnthropicLLM(_HTTPClient):
    def __init__(self, model, api_key, url=ANTHROPIC_URL, **kw):
        super().__init__(model, api_key, **kw)
        self.url = url

    def complete(self, system, user, temperature=0.0, max_tokens=1024):
        data = self._post(
            self.url,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
             "system": system, "messages": [{"role": "user", "content": user}]},
        )
        return "".join(b.get("text", "") for b in data.get("content", []))


class OpenAICompatLLM(_HTTPClient):
    def __init__(self, model, api_key, base_url, **kw):
        super().__init__(model, api_key, **kw)
        self.base_url = base_url.rstrip("/")

    def complete(self, system, user, temperature=0.0, max_tokens=1024):
        data = self._post(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
             "messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user}]},
        )
        return data["choices"][0]["message"]["content"] or ""


class CachedLLM:
    """Remembers deterministic (temperature 0) calls so re-checks are free and fast."""

    def __init__(self, inner):
        self.inner, self._cache = inner, {}

    def complete(self, system, user, temperature=0.0, max_tokens=1024):
        if temperature > 0:
            return self.inner.complete(system, user, temperature, max_tokens)
        key = (system, user, max_tokens)
        if key not in self._cache:
            self._cache[key] = self.inner.complete(system, user, temperature, max_tokens)
        return self._cache[key]


def make_llm(provider: Optional[str] = None, model: Optional[str] = None,
             api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Build a client from arguments or TRUTHCHECK_* environment variables."""
    provider = (provider or os.getenv("TRUTHCHECK_PROVIDER", "")).lower().strip()
    model = model or os.getenv("TRUTHCHECK_MODEL") or DEFAULT_MODELS.get(provider)
    api_key = api_key or os.getenv("TRUTHCHECK_API_KEY", "")
    base_url = base_url or os.getenv("TRUTHCHECK_BASE_URL")

    if provider == "demo":
        from .demo import DemoLLM
        return DemoLLM()
    if not provider:
        raise ValueError("No provider set. Use --provider or TRUTHCHECK_PROVIDER "
                         "(anthropic, openai, groq, gemini, ollama, openrouter, demo).")
    if not model:
        raise ValueError(f"No model set for '{provider}'. Use --model or TRUTHCHECK_MODEL.")
    if provider != "ollama" and not api_key:
        raise ValueError(f"No API key for '{provider}'. Set TRUTHCHECK_API_KEY.")
    if provider == "anthropic":
        return CachedLLM(AnthropicLLM(model, api_key, base_url or ANTHROPIC_URL))
    url = base_url or OPENAI_COMPAT_URLS.get(provider)
    if not url:
        raise ValueError(f"Unknown provider '{provider}'. Pass --base-url for a custom endpoint.")
    return CachedLLM(OpenAICompatLLM(model, api_key or "none", url))
