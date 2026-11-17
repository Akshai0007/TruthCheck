import json

import pytest

from eval.evaluate import score
from truthcheck.demo import DEMO_DOCS, DEMO_QUESTION, DemoLLM
from truthcheck.pipeline import TruthCheck, normalize_verdict, parse_json
from truthcheck.retrieval import (BM25, LocalDocsRetriever, WikipediaRetriever,
                                  make_passages, rank, tokenize)


class Scripted:
    """Returns a fixed reply per task tag, so tests can force specific model behaviour."""

    def __init__(self, **replies):
        self.replies = replies

    def complete(self, system, user, temperature=0.0, max_tokens=1024):
        for task, reply in self.replies.items():
            if f"[task:{task}]" in system:
                return reply(user) if callable(reply) else reply
        raise AssertionError("unexpected task: " + system[:40])


DOCS = LocalDocsRetriever(DEMO_DOCS)
ONE_CLAIM = json.dumps({"claims": [{"claim": "Marie Curie was born in Warsaw.",
                                    "sentence": "Marie Curie was born in Warsaw."}], "skipped": []})


# ---- building blocks -----------------------------------------------------
def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"a": 1} Hope that helps') == {"a": 1}
    assert parse_json('[{"x": 1}]') == [{"x": 1}]
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_normalize_verdict():
    assert normalize_verdict("Refuted") == "contradicted"
    assert normalize_verdict("NOT ENOUGH INFO") == "unverified"
    assert normalize_verdict("banana") == "unverified"


def test_bm25_ranks_relevant_passage_first():
    passages = make_passages("Paris is in France.\n\nBananas are yellow fruit.", "doc")
    top = rank("where is Paris", passages, k=2)
    assert "Paris" in top[0].text and len(top) == 1  # no-overlap passage is dropped
    assert BM25([]).scores(tokenize("x")) == []


def test_make_passages_windows_long_paragraphs():
    text = " ".join(f"Sentence number {i} here." for i in range(10))
    ps = make_passages(text, "d", window=3, stride=2)
    assert len(ps) > 1 and all(p.text for p in ps)


# ---- the full flow -------------------------------------------------------
def test_demo_end_to_end():
    tc = TruthCheck(DemoLLM(), DOCS, consistency_samples=1)
    rep = tc.ask_and_check(DEMO_QUESTION)
    assert [c.verdict for c in rep.claims] == ["supported", "supported", "contradicted"]
    assert rep.trust_score == 0.67
    assert "Fleming" in rep.corrected_answer
    assert {s["source"] for s in rep.sources} == {"Marie Curie", "Penicillin"}
    assert rep.claims[2].consistency == 1.0
    assert rep.counts() == {"supported": 2, "contradicted": 1, "unverified": 0}


def test_all_supported_returns_original_answer_unchanged():
    llm = Scripted(extract_claims=ONE_CLAIM, verify_claim=json.dumps(
        {"verdict": "supported", "reasoning": "ok", "evidence_ids": [1]}))
    rep = TruthCheck(llm, DOCS).check("Marie Curie was born in Warsaw.")
    assert rep.trust_score == 1.0 and rep.corrected_answer == rep.answer


# ---- safety guards -------------------------------------------------------
def test_verdict_without_cited_evidence_is_downgraded():
    llm = Scripted(extract_claims=ONE_CLAIM, correct_answer="(removed)", verify_claim=json.dumps(
        {"verdict": "supported", "reasoning": "I just know it", "evidence_ids": []}))
    rep = TruthCheck(llm, DOCS).check("Marie Curie was born in Warsaw.")
    assert rep.claims[0].verdict == "unverified"
    assert "Downgraded" in rep.claims[0].reasoning


def test_invented_evidence_ids_are_ignored():
    llm = Scripted(extract_claims=ONE_CLAIM, correct_answer="x", verify_claim=json.dumps(
        {"verdict": "supported", "reasoning": "r", "evidence_ids": [99]}))
    assert TruthCheck(llm, DOCS).check("a").claims[0].verdict == "unverified"


def test_garbage_from_checker_does_not_crash():
    llm = Scripted(extract_claims=ONE_CLAIM, verify_claim="lol", correct_answer="nothing left")
    rep = TruthCheck(llm, DOCS).check("Marie Curie was born in Warsaw.")
    assert rep.claims[0].verdict == "unverified"
    assert any("could not be checked" in n for n in rep.notes)


def test_extraction_failure_is_reported_not_raised():
    rep = TruthCheck(Scripted(extract_claims="???"), DOCS).check("Some text.")
    assert rep.claims == [] and any("Could not extract" in n for n in rep.notes)


def test_no_evidence_means_unverified():
    llm = Scripted(extract_claims=json.dumps({"claims": [{"claim": "Zorblax rules Mars."}]}),
                   correct_answer="")
    rep = TruthCheck(llm, DOCS).check("Zorblax rules Mars.")
    assert rep.claims[0].verdict == "unverified" and "No evidence" in rep.claims[0].reasoning
    assert rep.corrected_answer == "No statement in this answer could be verified."


def test_refusal_is_skipped():
    rep = TruthCheck(Scripted(), DOCS).check("I don't know the answer to that.")
    assert not rep.claims and rep.notes


def test_invalid_citation_markers_are_stripped():
    llm = Scripted(extract_claims=ONE_CLAIM, verify_claim=json.dumps(
        {"verdict": "unverified", "reasoning": "r", "evidence_ids": []}),
        correct_answer="Born in Warsaw [1][7].")
    rep = TruthCheck(llm, DOCS).check("Marie Curie was born in Warsaw.")
    assert "[7]" not in rep.corrected_answer and "[1]" not in rep.corrected_answer  # no sources exist


def test_judge_can_request_a_second_search():
    seen_queries = []

    class Spy(LocalDocsRetriever):
        def search(self, query, k=4):
            seen_queries.append(query)
            return super().search(query, k)

    replies = iter([
        json.dumps({"verdict": "unverified", "reasoning": "r", "evidence_ids": [], "next_query": "Penicillin Fleming"}),
        json.dumps({"verdict": "supported", "reasoning": "r", "evidence_ids": [1]}),
    ])
    llm = Scripted(verify_claim=lambda u: next(replies))
    res = TruthCheck(llm, Spy(DEMO_DOCS), max_search_steps=2).verify_claim("Fleming found penicillin")
    assert seen_queries == ["Fleming found penicillin", "Penicillin Fleming"]
    assert res.verdict == "supported"


# ---- Wikipedia parsing (network faked; live call must be tested on your machine) ----
class FakeSession:
    def __init__(self):
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        class R:
            def raise_for_status(self): pass
            def json(_):
                if params.get("list") == "search":
                    return {"query": {"search": [{"title": "Marie Curie"}]}}
                return {"query": {"pages": [{"title": "Marie Curie",
                                             "extract": DEMO_DOCS["Marie Curie"]}]}}
        return R()


def test_wikipedia_retriever_parses_api_shape():
    out = WikipediaRetriever(session=FakeSession()).search("Where was Marie Curie born?")
    assert out and out[0].source == "Wikipedia: Marie Curie"
    assert out[0].url == "https://en.wikipedia.org/wiki/Marie_Curie"


# ---- evaluation maths ------------------------------------------------------
def test_score_metrics():
    g = ["SUPPORTS", "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    p = ["SUPPORTS", "REFUTES", "REFUTES", "NOT ENOUGH INFO"]
    s = score(g, p)
    assert s["accuracy"] == 0.75
    assert s["per_label"]["REFUTES"]["precision"] == 0.5 and s["per_label"]["REFUTES"]["recall"] == 1.0
    with pytest.raises(ValueError):
        score([], [])
