"""Measure how good the verifier is on labelled claims (FEVER-style).

Input: a .jsonl file, one object per line:  {"claim": "...", "label": "SUPPORTS|REFUTES|NOT ENOUGH INFO"}
FEVER (https://fever.ai) uses exactly these labels, so you can convert a slice of it into
this format and run:

  python -m eval.evaluate eval/sample_claims.jsonl --provider groq --model <name> --wikipedia --limit 100

Long runs: every finished claim is saved at once to --out (default eval/results/last_run.jsonl) and a progress
line is printed per claim. Stop any time with Ctrl+C (metrics are computed from what finished), continue with
--resume, or recompute metrics without any API calls with --summarize.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
VERDICT_TO_LABEL = {"supported": "SUPPORTS", "contradicted": "REFUTES", "unverified": "NOT ENOUGH INFO"}


def score(golds: list, preds: list) -> dict:
    """Accuracy, per-label precision/recall/F1 and a confusion matrix."""
    if len(golds) != len(preds) or not golds:
        raise ValueError("golds and preds must be non-empty and the same length")
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(golds, preds):
        confusion[g][p] += 1
    per_label = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        pred_n = sum(confusion[g][lab] for g in LABELS)
        gold_n = sum(confusion[lab].values())
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / gold_n if gold_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_label[lab] = {"precision": round(prec, 3), "recall": round(rec, 3),
                          "f1": round(f1, 3), "n": gold_n}
    acc = sum(confusion[l][l] for l in LABELS) / len(golds)
    return {"accuracy": round(acc, 3), "per_label": per_label, "confusion": confusion, "n": len(golds)}


def load(path: str, limit: int = 0, seed: int = 0) -> list:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for r in rows:
        if r["label"] not in LABELS:
            raise ValueError(f"Unknown label {r['label']!r}; use one of {LABELS}")
    if limit and limit < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def run(rows: list, tc) -> dict:
    golds, preds, errors = [], [], 0
    for r in rows:
        res = tc.verify_claim(r["claim"])
        errors += "Checker error" in res.reasoning
        golds.append(r["label"])
        preds.append(VERDICT_TO_LABEL[res.verdict])
    out = score(golds, preds)
    out["checker_errors"] = errors
    return out


def run_streaming(rows: list, tc, out_path, workers: int = 3, resume: bool = False, log=sys.stderr) -> None:
    """Check claims in parallel; append each finished claim to out_path immediately (so nothing is lost)."""
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    done = set(_read_results(p)) if resume else set()
    if not resume:
        p.write_text("", encoding="utf-8")
    todo = [r for r in rows if r["claim"] not in done]
    total, lock, finished, failed = len(rows), threading.Lock(), [len(rows) - len(todo)], [0]

    def work(r):
        t0 = time.time()
        try:
            res = tc.verify_claim(r["claim"])
            pred, why = VERDICT_TO_LABEL[res.verdict], res.reasoning
        except Exception as e:   # never lose a long run to one bad claim
            pred, why = "NOT ENOUGH INFO", f"Checker error: {e}"
        return {"claim": r["claim"], "label": r["label"], "pred": pred, "error": "Checker error" in why,
                "why": why[:300], "seconds": round(time.time() - t0, 1)}

    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = [ex.submit(work, r) for r in todo]
    try:
        for f in as_completed(futures):
            d = f.result()
            with lock:
                with p.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                finished[0] += 1
                failed[0] = failed[0] + 1 if d["error"] else 0
                print(f"[{finished[0]}/{total}] gold={d['label']:<15} pred={d['pred']:<15}"
                      f"{' ERROR' if d['error'] else ''} ({d['seconds']}s)", file=log, flush=True)
            if failed[0] >= 5:   # five failures in a row: every claim would fail, stop wasting time
                ex.shutdown(wait=False, cancel_futures=True)
                sys.exit(f"Stopped: 5 claims in a row failed. Last error: {d['why']}")
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        print("\nStopped early. Metrics below use the claims finished so far.", file=log, flush=True)
    else:
        ex.shutdown(wait=True)


def _read_results(path) -> dict:
    out = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                out[d["claim"]] = d
            except (ValueError, KeyError):
                continue
    return out


def summarize(rows: list, out_path) -> dict:
    """Metrics from the saved results of the claims in `rows` (no API calls)."""
    saved = _read_results(out_path)
    got = [saved[r["claim"]] for r in rows if r["claim"] in saved]
    if not got:
        sys.exit("No finished claims found yet. Run the evaluation first.")
    out = score([d["label"] for d in got], [d["pred"] for d in got])
    out["checker_errors"] = sum(d["error"] for d in got)
    out["requested"] = len(rows)
    if len(got) < len(rows):
        out["note"] = f"partial: {len(got)} of {len(rows)} claims finished"
    return out


def main(argv=None):
    from truthcheck.cli import LocalDocsRetriever, WikipediaRetriever
    from truthcheck.llm import make_llm
    from truthcheck.pipeline import TruthCheck

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--provider"); ap.add_argument("--model")
    ap.add_argument("--docs"); ap.add_argument("--wikipedia", action="store_true")
    ap.add_argument("--max-search-steps", type=int, default=2,
                    help="1 = no second search, 2 = one retry (default)")
    ap.add_argument("--workers", type=int, default=3, help="claims checked at the same time (default 3)")
    ap.add_argument("--out", default="eval/results/last_run.jsonl", help="where finished claims are saved")
    ap.add_argument("--resume", action="store_true", help="skip claims already finished in --out")
    ap.add_argument("--summarize", action="store_true", help="only compute metrics from --out, no API calls")
    a = ap.parse_args(argv)
    rows = load(a.path, a.limit)
    llm = None
    if not a.summarize:
        try:
            llm = make_llm(a.provider, a.model)
        except ValueError as e:
            sys.exit(f"Setup problem: {e}")
        retr = LocalDocsRetriever.from_folder(a.docs) if a.docs else WikipediaRetriever()
        run_streaming(rows, TruthCheck(llm, retr, max_search_steps=a.max_search_steps), a.out,
                      workers=a.workers, resume=a.resume)
    result = summarize(rows, a.out)
    result["models_used"] = getattr(getattr(llm, "inner", None), "usage", None)   # which models answered
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
