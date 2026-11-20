import json

from eval import evaluate, make_fever_sample as mfs
from truthcheck.demo import DEMO_DOCS
from truthcheck.pipeline import TruthCheck
from truthcheck.retrieval import LocalDocsRetriever


def _fever_file(tmp_path):
    rows = ([{"claim": f"s{i}", "label": "SUPPORTS", "evidence": []} for i in range(5)]
            + [{"claim": f"r{i}", "label": "REFUTES"} for i in range(5)]
            + [{"claim": f"n{i}", "label": "NOT ENOUGH INFO"} for i in range(5)]
            + [{"claim": "bad", "label": "SOMETHING ELSE"}])
    p = tmp_path / "fever.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_fever_sample_is_balanced_and_valid(tmp_path):
    out = tmp_path / "out.jsonl"
    mfs.main([str(_fever_file(tmp_path)), str(out), "--per-label", "3"])
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 9
    assert {r["label"] for r in rows} == set(mfs.LABELS)
    assert all(set(r) == {"claim", "label"} for r in rows)
    assert len(evaluate.load(str(out))) == 9   # evaluate.py accepts the file


def test_evaluate_accepts_max_search_steps_flag(tmp_path, capsys):
    p = tmp_path / "docs"; p.mkdir()
    (p / "a.txt").write_text(DEMO_DOCS["Marie Curie"], encoding="utf-8")
    evaluate.main(["eval/sample_claims.jsonl", "--provider", "demo", "--docs", str(p),
                   "--max-search-steps", "1", "--out", str(tmp_path / "r.jsonl")])
    assert '"accuracy"' in capsys.readouterr().out


class _FakeTC:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def verify_claim(self, claim):
        from truthcheck.models import ClaimResult
        self.calls += 1
        r = ClaimResult(claim=claim, sentence="")
        r.verdict, r.reasoning = ("unverified", "Checker error: boom") if self.fail else ("supported", "ok")
        return r


def _rows(n):
    return [{"claim": f"c{i}", "label": "SUPPORTS"} for i in range(n)]


def test_streaming_saves_each_claim_and_resume_skips_finished(tmp_path):
    out, log = tmp_path / "o.jsonl", open(tmp_path / "log.txt", "w")
    evaluate.run_streaming(_rows(4), _FakeTC(), out, workers=2, log=log)
    assert len(out.read_text().splitlines()) == 4
    tc = _FakeTC()
    evaluate.run_streaming(_rows(6), tc, out, workers=2, resume=True, log=log)
    assert tc.calls == 2                                   # only the 2 new claims were checked
    summary = evaluate.summarize(_rows(6), out)
    assert summary["accuracy"] == 1.0 and summary["n"] == 6 and "note" not in summary


def test_summarize_reports_partial_runs(tmp_path):
    out, log = tmp_path / "o.jsonl", open(tmp_path / "log.txt", "w")
    evaluate.run_streaming(_rows(3), _FakeTC(), out, log=log)
    s = evaluate.summarize(_rows(10), out)
    assert s["n"] == 3 and "partial" in s["note"]


def test_five_failures_in_a_row_stop_the_run(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="5 claims in a row"):
        evaluate.run_streaming(_rows(30), _FakeTC(fail=True), tmp_path / "o.jsonl", workers=1,
                               log=open(tmp_path / "log.txt", "w"))
