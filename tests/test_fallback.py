import pytest

from truthcheck import llm as L
from truthcheck.llm import FallbackLLM, LLMError, make_llm


class Fake:
    def __init__(self, model, outcome):
        self.model, self.outcome, self.calls = model, outcome, 0

    def complete(self, system, user, temperature=0.0, max_tokens=10):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)


def test_moves_to_next_model_when_one_is_busy_or_missing():
    a = Fake("a", LLMError("busy", status=429))
    b = Fake("b", LLMError("no such model", status=404))
    c = Fake("c", "answer")
    chain = FallbackLLM([a, b, c])
    assert chain.complete("s", "u") == "answer"
    assert chain.usage == {"c": 1}


def test_failed_model_is_skipped_for_the_next_calls():
    a, b = Fake("a", LLMError("busy", status=429)), Fake("b", "ok")
    chain = FallbackLLM([a, b])
    chain.complete("s", "u"); chain.complete("s", "u2")
    assert a.calls == 1 and b.calls == 2     # a was only tried once, then rested


def test_wrong_api_key_stops_immediately():
    a, b = Fake("a", LLMError("bad key", status=401, fatal=True)), Fake("b", "ok")
    with pytest.raises(LLMError, match="bad key"):
        FallbackLLM([a, b]).complete("s", "u")
    assert b.calls == 0


def test_all_models_failing_lists_every_reason():
    chain = FallbackLLM([Fake("a", LLMError("busy", status=429)), Fake("b", LLMError("empty reply"))])
    with pytest.raises(LLMError) as e:
        chain.complete("s", "u")
    assert "a: busy" in str(e.value) and "b: empty reply" in str(e.value)


def test_make_llm_uses_provider_key_then_generic_key(monkeypatch):
    monkeypatch.setenv("TRUTHCHECK_API_KEY", "generic")
    assert L.find_api_key("groq") == "generic"
    monkeypatch.setenv("GROQ_API_KEY", "specific")
    assert L.find_api_key("groq") == "specific"
    monkeypatch.setenv("GEMINI_MODELS", "m1, m2")
    assert L.models_for("gemini") == ["m1", "m2"]
    chain = make_llm("groq").inner
    assert [c.model for c in chain.clients] == L.DEFAULT_CHAINS["groq"]
    assert all(c.api_key == "specific" for c in chain.clients)
    assert make_llm("groq", "my-model").inner.model == "my-model"   # pinned: no fallback


def test_missing_key_message_names_the_variable():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        make_llm("groq")


def test_openai_payload_uses_new_style_fields(monkeypatch):
    sent = {}

    class R:
        status_code = 200
        headers = {}
        def json(self): return {"choices": [{"message": {"content": "hi"}}]}

    monkeypatch.setattr(L.requests, "post", lambda url, headers, json, timeout: sent.update(json) or R())
    L.OpenAICompatLLM("gpt-x", "k", "https://api.openai.com/v1").complete("s", "u", 0.0, 99)
    assert sent["max_completion_tokens"] == 99 and "temperature" not in sent and "max_tokens" not in sent
    L.OpenAICompatLLM("m", "k", "https://api.groq.com/openai/v1").complete("s", "u", 0.0, 99)
    assert sent["max_tokens"] == 99 and sent["temperature"] == 0.0


def test_wikipedia_retries_when_rate_limited(monkeypatch):
    from truthcheck.retrieval import WikipediaRetriever
    monkeypatch.setattr("truthcheck.retrieval.time.sleep", lambda s: None)
    codes = iter([429, 429, 200])

    class R:
        def __init__(self, code): self.status_code, self.headers = code, {}
        def raise_for_status(self): pass
        def json(self): return {"query": {"search": []}}

    class S:
        headers = {}
        def get(self, *a, **k): return R(next(codes))

    assert WikipediaRetriever(session=S()).search("x") == []   # 3rd try succeeds, no error raised
