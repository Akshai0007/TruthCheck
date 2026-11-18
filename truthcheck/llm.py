"""Tiny LLM layer. Any object with .complete(system, user, temperature, max_tokens) works.

Supported out of the box:
  groq / openai / gemini / ollama / openrouter -> OpenAI-compatible endpoints
  anthropic                                    -> Anthropic Messages API
  demo                                         -> scripted offline stand-in (sample input only)

Keys come from the environment / .env file: GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
(ANTHROPIC_API_KEY, OPENROUTER_API_KEY), with TRUTHCHECK_API_KEY as a fallback for any provider.

Models are chosen automatically. Each provider has an ordered list (DEFAULT_CHAINS). If a model is
busy, rate-limited, unavailable or returns nothing, the next one is tried until an answer comes back.
Override an order without touching code:  GROQ_MODELS=model-a,model-b  (same for OPENAI_/GEMINI_).
Model catalogues change often; check each provider's docs and update DEFAULT_CHAINS when needed.
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

# Preferred model first. Checked against each provider's model docs in November 2026.
DEFAULT_CHAINS = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "openai": ["gpt-5.4-mini", "gpt-6-luna", "gpt-5.6-luna", "gpt-5.4-nano"],
    "gemini": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
               "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
    "anthropic": ["claude-haiku-5-5", "claude-sonnet-5-5"],
    "ollama": ["llama3.1"],
}
DEFAULT_MODELS = {p: c[0] for p, c in DEFAULT_CHAINS.items()}   # kept for older code
KEY_ENV = {
    "groq": ["GROQ_API_KEY"], "openai": ["OPENAI_API_KEY"], "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"], "openrouter": ["OPENROUTER_API_KEY"],
}
DEFAULT_MAX_TOKENS = 3000   # reasoning models spend part of this on hidden thinking


class LLMError(RuntimeError):
    """A failed model call. `fatal` means retrying other models cannot help (wrong API key)."""

    def __init__(self, message, status=None, fatal=False):
        super().__init__(message)
        self.status, self.fatal = status, fatal


class LLM(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> str: ...


def find_api_key(provider: str) -> str:
    """Provider-specific variable first (GROQ_API_KEY ...), then the generic TRUTHCHECK_API_KEY."""
    for name in KEY_ENV.get(provider, []) + ["TRUTHCHECK_API_KEY"]:
        v = os.getenv(name, "").strip()
        if v:
            return v
    return ""


def models_for(provider: str) -> list:
    """The ordered model list for a provider: GROQ_MODELS-style override, else the built-in chain."""
    raw = os.getenv(f"{provider.upper()}_MODELS", "")
    custom = [m.strip() for m in raw.split(",") if m.strip()]
    return custom or list(DEFAULT_CHAINS.get(provider, []))


class _HTTPClient:
    def __init__(self, model: str, api_key: str, timeout: int = 60, retries: int = 3):
        self.model, self.api_key, self.timeout, self.retries = model, api_key, timeout, retries

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        last, status = "no attempt made", None
        for attempt in range(self.retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last, status = str(e), None
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 529):  # busy / rate-limited: wait, then retry
                last, status = f"HTTP {r.status_code}: {r.text[:200]}", r.status_code
                if attempt < self.retries - 1:
                    wait = r.headers.get("Retry-After", "")
                    time.sleep(min(float(wait), 30) if wait.replace(".", "", 1).isdigit() else 2 ** attempt)
                continue
            if r.status_code >= 400:  # wrong key, wrong model name...: retrying the same call cannot help
                raise LLMError(f"LLM request rejected (HTTP {r.status_code}): {r.text[:300]}",
                               status=r.status_code, fatal=r.status_code == 401)
            try:
                return r.json()
            except ValueError:
                raise LLMError("LLM reply was not valid JSON", status=r.status_code)
        raise LLMError(f"LLM request failed after {self.retries} tries: {last}", status=status)


class AnthropicLLM(_HTTPClient):
    def __init__(self, model, api_key, url=ANTHROPIC_URL, **kw):
        super().__init__(model, api_key, **kw)
        self.url = url

    def complete(self, system, user, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS):
        data = self._post(
            self.url,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
             "system": system, "messages": [{"role": "user", "content": user}]},
        )
        text = "".join(b.get("text", "") for b in data.get("content", []))
        if not text.strip():
            raise LLMError(f"{self.model} returned an empty reply")
        return text


class OpenAICompatLLM(_HTTPClient):
    def __init__(self, model, api_key, base_url, **kw):
        super().__init__(model, api_key, **kw)
        self.base_url = base_url.rstrip("/")
        self.openai_style = "api.openai.com" in self.base_url

    def complete(self, system, user, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.openai_style:   # newer OpenAI models want max_completion_tokens and only the default temperature
            payload["max_completion_tokens"] = max_tokens
        else:
            payload.update(max_tokens=max_tokens, temperature=temperature)
        data = self._post(f"{self.base_url}/chat/completions",
                          {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{self.model} sent an unexpected reply format")
        if not text.strip():
            raise LLMError(f"{self.model} returned an empty reply")
        return text


class FallbackLLM:
    """Tries models in order; moves to the next when one is busy, rate-limited, missing or empty.

    A model that just failed is skipped for a while (60 s if busy, 10 min if unavailable) so the next
    claims do not waste time on it. `usage` counts how many answers each model gave.
    """

    def __init__(self, clients, rounds: int = 2, pause: float = 6.0):
        self.clients, self.rounds, self.pause = list(clients), rounds, pause
        self.usage, self._cool = {}, {}

    def complete(self, system, user, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS):
        errors = []
        for rnd in range(self.rounds):
            errors = []
            ready = [c for c in self.clients if self._cool.get(c.model, 0) <= time.time()]
            for c in ready or self.clients:
                try:
                    out = c.complete(system, user, temperature, max_tokens)
                except LLMError as e:
                    if e.fatal:      # e.g. wrong API key: every model would fail the same way
                        raise
                    errors.append(f"{c.model}: {e}")
                    self._cool[c.model] = time.time() + (600 if e.status in (400, 403, 404) else 60)
                    continue
                self.usage[c.model] = self.usage.get(c.model, 0) + 1
                return out
            if rnd < self.rounds - 1:
                time.sleep(self.pause)
        raise LLMError("All models failed - " + " | ".join(errors)[:700])


class CachedLLM:
    """Remembers deterministic (temperature 0) calls so re-checks are free and fast."""

    def __init__(self, inner):
        self.inner, self._cache = inner, {}

    def complete(self, system, user, temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS):
        if temperature > 0:
            return self.inner.complete(system, user, temperature, max_tokens)
        key = (system, user, max_tokens)
        if key not in self._cache:
            self._cache[key] = self.inner.complete(system, user, temperature, max_tokens)
        return self._cache[key]


def make_llm(provider: Optional[str] = None, model: Optional[str] = None,
             api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Build a client. With no `model`, the provider's automatic fallback list is used.
    Passing `model` pins that single model (handy for repeatable evaluation runs)."""
    provider = (provider or os.getenv("TRUTHCHECK_PROVIDER", "")).lower().strip()
    base_url = base_url or os.getenv("TRUTHCHECK_BASE_URL")

    if provider == "demo":
        from .demo import DemoLLM
        return DemoLLM()
    if not provider:
        raise ValueError("No provider set. Choose one in the app, use --provider, or set TRUTHCHECK_PROVIDER "
                         "(groq, openai, gemini, anthropic, ollama, openrouter, demo).")
    api_key = api_key or find_api_key(provider)
    if provider != "ollama" and not api_key:
        var = (KEY_ENV.get(provider) or ["TRUTHCHECK_API_KEY"])[0]
        raise ValueError(f"No API key for '{provider}'. Add {var}=... to your .env file (see .env.example).")
    pinned = bool(model)
    models = [model] if pinned else models_for(provider)
    if not models:
        raise ValueError(f"No models known for '{provider}'. Use --model, or set {provider.upper()}_MODELS.")
    kw = {} if pinned else {"retries": 1}   # in a chain, a failing model hands over quickly
    if provider == "anthropic":
        clients = [AnthropicLLM(m, api_key, base_url or ANTHROPIC_URL, **kw) for m in models]
    else:
        url = base_url or OPENAI_COMPAT_URLS.get(provider)
        if not url:
            raise ValueError(f"Unknown provider '{provider}'. Pass --base-url for a custom endpoint.")
        clients = [OpenAICompatLLM(m, api_key or "none", url, **kw) for m in models]
    return CachedLLM(clients[0] if len(clients) == 1 else FallbackLLM(clients))
