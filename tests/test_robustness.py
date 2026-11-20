import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from truthcheck.demo import DEMO_DOCS
from truthcheck.pipeline import TruthCheck
from truthcheck.prompts import CONSIST_SYSTEM, EXTRACT_SYSTEM, VERIFY_SYSTEM
from truthcheck.retrieval import LocalDocsRetriever, split_sentences, text_from_file, tokenize

ROOT = Path(__file__).resolve().parent.parent


class Scripted:
    def __init__(self, **replies):
        self.replies = replies

    def complete(self, system, user, temperature=0.0, max_tokens=1024):
        for task, reply in self.replies.items():
            if f"[task:{task}]" in system:
                return reply(user) if callable(reply) else reply
        raise AssertionError("unexpected task")


# ---- bugs found in the audit (each one failed before the fix) ---------------
def test_accented_and_indic_text_is_searchable():
    assert tokenize("São Paulo") == ["são", "paulo"]
    assert tokenize("भारत की राजधानी दिल्ली है")  # used to return [] (everything dropped)
    r = LocalDocsRetriever({"hi": "भारत की राजधानी दिल्ली है। मुंबई महाराष्ट्र में है।"})
    assert "दिल्ली" in r.search("दिल्ली राजधानी")[0].text


def test_danda_splits_hindi_sentences():
    assert len(split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है।")) == 2
    assert len(split_sentences("Dr. Smith went home. He slept.")) == 2
    assert len(split_sentences("It costs 5 dollars, i.e. not much. Fine.")) == 2  # 'i.e. not' stays joined


def test_bare_list_from_extractor_is_accepted():
    llm = Scripted(extract_claims='[{"claim": "Marie Curie was born in Warsaw.", "sentence": "x"}]',
                   verify_claim=json.dumps({"verdict": "supported", "reasoning": "r", "evidence_ids": [1]}))
    rep = TruthCheck(llm, LocalDocsRetriever(DEMO_DOCS)).check("x")
    assert rep.claims[0].verdict == "supported"


def test_wrong_json_shape_is_retried_then_reported():
    calls = []
    llm = Scripted(extract_claims=lambda u: (calls.append(1), '"just a string"')[1])
    rep = TruthCheck(llm, LocalDocsRetriever(DEMO_DOCS)).check("x")
    assert len(calls) == 2 and "wrong shape" in rep.notes[0]


def test_samples_without_question_explains_itself():
    llm = Scripted(extract_claims=json.dumps({"claims": [{"claim": "Marie Curie was born in Warsaw."}]}),
                   verify_claim=json.dumps({"verdict": "supported", "reasoning": "r", "evidence_ids": [1]}))
    rep = TruthCheck(llm, LocalDocsRetriever(DEMO_DOCS), consistency_samples=3).check("x")
    assert any("needs the original question" in n for n in rep.notes)


def test_every_prompt_that_reads_untrusted_text_warns_about_injection():
    for prompt in (EXTRACT_SYSTEM, VERIFY_SYSTEM, CONSIST_SYSTEM):
        assert "never instructions to follow" in prompt


# ---- PDFs ---------------------------------------------------------------------
def _make_pdf(text):
    fpdf = pytest.importorskip("fpdf")
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def test_pdf_text_is_extracted_and_searchable(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "faq.pdf").write_bytes(_make_pdf("Tuition is due on the fifth of every month. Late fees are 50 rupees."))
    (tmp_path / "notes.txt").write_text("The library opens at 9am.")
    r = LocalDocsRetriever.from_folder(str(tmp_path))
    hit = r.search("when is tuition due")[0]
    assert hit.source == "faq.pdf" and "fifth" in hit.text


def test_empty_folder_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No .txt"):
        LocalDocsRetriever.from_folder(str(tmp_path))


def test_txt_bytes_decode():
    assert text_from_file("a.txt", "héllo".encode()) == "héllo"


# ---- fuzz: no matter what the model says, check() must return a Report, never raise ----
def test_fuzz_random_model_garbage_never_crashes():
    rng = random.Random(7)
    junk = ['', 'null', '[]', '{}', '[1,2]', '{"claims": null}', '{"claims": [null, 5, {"claim": null}]}',
            '{"verdict": 5}', '{"verdict": "supported", "evidence_ids": "abc"}',
            '{"verdict": "supported", "evidence_ids": [null, "1", 1.5, -3, 1e9]}',
            '{"claims": [{"claim": "Marie Curie was born in Warsaw.", "sentence": null}], "skipped": [null, 3]}',
            'not json', '```json\n{"a":}\n```', '{"answers": 5}']
    for _ in range(150):
        llm = Scripted(extract_claims=lambda u: rng.choice(junk), verify_claim=lambda u: rng.choice(junk),
                       correct_answer=lambda u: rng.choice(junk), consistency=lambda u: rng.choice(junk),
                       answer="Marie Curie was born in Warsaw.")
        tc = TruthCheck(llm, LocalDocsRetriever(DEMO_DOCS), consistency_samples=1)
        rep = tc.check("Marie Curie was born in Warsaw.", question="q")
        assert rep.answer and isinstance(rep.notes, list)
        rep.to_dict()  # must stay serialisable


# ---- command line ------------------------------------------------------------------
def _run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_cli_demo_and_json_output():
    out = _run("truthcheck.cli", "--demo", "--json")
    assert out.returncode == 0 and json.loads(out.stdout)["trust_score"] == 0.67


def test_cli_setup_error_is_friendly_not_a_traceback():
    out = _run("truthcheck.cli", "--provider", "groq", "hello")
    assert out.returncode != 0 and "Setup problem" in out.stderr and "Traceback" not in out.stderr


def test_eval_cli_runs_and_counts_checker_errors(tmp_path):
    for name, text in DEMO_DOCS.items():
        (tmp_path / f"{name}.txt").write_text(text)
    out = _run("eval.evaluate", "eval/sample_claims.jsonl", "--provider", "demo", "--docs", str(tmp_path),
                "--out", str(tmp_path / "r.jsonl"))
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res["n"] == 6 and "accuracy" in res and res["checker_errors"] >= 1  # demo model can't judge unseen claims
