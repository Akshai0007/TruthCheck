"""Measure how good the verifier is on labelled claims (FEVER-style).

Input: a .jsonl file, one object per line:  {"claim": "...", "label": "SUPPORTS|REFUTES|NOT ENOUGH INFO"}
FEVER (https://fever.ai) uses exactly these labels, so you can convert a slice of it into
this format and run:

  python -m eval.evaluate eval/sample_claims.jsonl --provider groq --model <name> --wikipedia --limit 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys

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


def main(argv=None):
    from truthcheck.cli import LocalDocsRetriever, WikipediaRetriever
    from truthcheck.llm import make_llm
    from truthcheck.pipeline import TruthCheck

    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--provider"); ap.add_argument("--model")
    ap.add_argument("--docs"); ap.add_argument("--wikipedia", action="store_true")
    a = ap.parse_args(argv)
    try:
        llm = make_llm(a.provider, a.model)
    except ValueError as e:
        sys.exit(f"Setup problem: {e}")
    retr = LocalDocsRetriever.from_folder(a.docs) if a.docs else WikipediaRetriever()
    print(json.dumps(run(load(a.path, a.limit), TruthCheck(llm, retr)), indent=2))


if __name__ == "__main__":
    main()
