"""Runs the REAL network code (requests, headers, retries, JSON shapes) against local fake servers.

The fake servers answer with DemoLLM / DEMO_DOCS, so the whole pipeline is exercised through HTTP
for both LLM styles and for the Wikipedia API shape. This proves our side of each protocol;
it cannot prove the real providers behave identically, which is why a live smoke test is still needed.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from truthcheck.demo import DEMO_ANSWER, DEMO_DOCS, DemoLLM
from truthcheck.llm import make_llm
from truthcheck.pipeline import TruthCheck
from truthcheck.retrieval import WikipediaRetriever

STATE = {"seen": [], "fail_next": 0, "reject": None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, headers=None):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["seen"].append((self.path, dict(self.headers), body))
        if STATE["reject"]:
            return self._send(STATE["reject"], {"error": {"message": "invalid api key"}})
        if STATE["fail_next"] > 0:
            STATE["fail_next"] -= 1
            return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
        if self.path == "/anthropic/v1/messages":
            assert self.headers["x-api-key"] == "k-anthropic"
            assert self.headers["anthropic-version"]
            text = DemoLLM().complete(body["system"], body["messages"][0]["content"])
            return self._send(200, {"content": [{"type": "text", "text": text}]})
        if self.path == "/openai/v1/chat/completions":
            assert self.headers["Authorization"] == "Bearer k-openai"
            sysmsg, usermsg = body["messages"][0], body["messages"][1]
            assert sysmsg["role"] == "system" and usermsg["role"] == "user"
            text = DemoLLM().complete(sysmsg["content"], usermsg["content"])
            return self._send(200, {"choices": [{"message": {"content": text}}]})
        self._send(404, {})

    def do_GET(self):
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        assert "truthcheck" in self.headers["User-Agent"]
        assert q["format"] == "json" and q["formatversion"] == "2"
        if q.get("list") == "search":
            return self._send(200, {"query": {"search": [{"title": t} for t in DEMO_DOCS]}})
        title = q["titles"]
        return self._send(200, {"query": {"pages": [{"title": title, "extract": DEMO_DOCS[title]}]}})


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture(autouse=True)
def reset_state():
    STATE.update(seen=[], fail_next=0, reject=None)


def _wiki(server):
    return WikipediaRetriever(api_url=server + "/w/api.php")


def test_full_pipeline_over_http_anthropic_style_plus_wikipedia_api(server):
    llm = make_llm("anthropic", "some-model", "k-anthropic", base_url=server + "/anthropic/v1/messages")
    rep = TruthCheck(llm, _wiki(server)).check(DEMO_ANSWER)
    assert [c.verdict for c in rep.claims] == ["supported", "supported", "contradicted"]
    assert rep.sources[0]["url"].startswith("https://en.wikipedia.org/wiki/")
    path, headers, body = STATE["seen"][0]
    assert body["model"] == "some-model" and "temperature" in body and body["max_tokens"] >= 1024


def test_full_pipeline_over_http_openai_compatible_style(server):
    llm = make_llm("groq", "m", "k-openai", base_url=server + "/openai/v1")
    rep = TruthCheck(llm, _wiki(server)).check(DEMO_ANSWER)
    assert rep.trust_score == 0.67 and "Fleming" in rep.corrected_answer


def test_rate_limit_is_retried_and_succeeds(server):
    llm = make_llm("groq", "m", "k-openai", base_url=server + "/openai/v1")
    STATE["fail_next"] = 2
    assert "Warsaw" in llm.complete("[task:answer]", "q")
    assert len(STATE["seen"]) == 3  # two 429s then success


def test_bad_key_fails_fast_with_clear_message(server):
    llm = make_llm("groq", "m", "wrong", base_url=server + "/openai/v1")
    STATE["reject"] = 401
    t0 = time.time()
    with pytest.raises(RuntimeError, match=r"HTTP 401.*invalid api key"):
        llm.complete("[task:answer]", "q")
    assert time.time() - t0 < 1.0 and len(STATE["seen"]) == 1  # no pointless retries


def test_bad_key_during_check_shows_up_as_a_report_note(server):
    llm = make_llm("groq", "m", "wrong", base_url=server + "/openai/v1")
    STATE["reject"] = 401
    rep = TruthCheck(llm, _wiki(server)).check("Anything at all.")
    assert any("HTTP 401" in n for n in rep.notes)


def test_unreachable_server_gives_clear_error():
    llm = make_llm("ollama", "llama3.1", base_url="http://127.0.0.1:9/v1")  # nothing listens on port 9
    llm.inner.retries = 1
    with pytest.raises(RuntimeError, match="failed after"):
        llm.complete("s", "u")


def test_cache_avoids_repeat_calls_but_not_for_sampling(server):
    llm = make_llm("groq", "m", "k-openai", base_url=server + "/openai/v1")
    llm.complete("[task:answer]", "q"); llm.complete("[task:answer]", "q")
    assert len(STATE["seen"]) == 1
    llm.complete("[task:answer]", "q", temperature=1.0); llm.complete("[task:answer]", "q", temperature=1.0)
    assert len(STATE["seen"]) == 3


def test_setup_errors_are_explained():
    with pytest.raises(ValueError, match="No provider"):
        make_llm("")
    with pytest.raises(ValueError, match="API key"):
        make_llm("groq", "m", "")
    with pytest.raises(ValueError, match="No model"):
        make_llm("groq", None, "k")
    assert make_llm("ollama", None, "")  # local models need neither key nor explicit model
