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
                   "--max-search-steps", "1"])
    assert '"accuracy"' in capsys.readouterr().out
