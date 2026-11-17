"""Turn a FEVER download into the small claims file that eval/evaluate.py reads.

FEVER's labelled files are .jsonl with at least {"claim": ..., "label": ...}, where label is
SUPPORTS, REFUTES or NOT ENOUGH INFO -- exactly the labels this project uses.

  python -m eval.make_fever_sample shared_task_dev.jsonl eval/fever_150.jsonl --per-label 50
"""
from __future__ import annotations

import argparse
import json
import random

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def sample(rows: list, per_label: int, seed: int = 0) -> list:
    """Balanced random sample: up to `per_label` claims for each of the three labels."""
    rng = random.Random(seed)
    out = []
    for lab in LABELS:
        pool = [r for r in rows if r.get("label") == lab and r.get("claim")]
        rng.shuffle(pool)
        out += [{"claim": r["claim"].strip(), "label": lab} for r in pool[:per_label]]
    rng.shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fever_file"); ap.add_argument("out_file")
    ap.add_argument("--per-label", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.fever_file, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    chosen = sample(rows, a.per_label, a.seed)
    if not chosen:
        raise SystemExit("No claims with labels SUPPORTS / REFUTES / NOT ENOUGH INFO were found.")
    with open(a.out_file, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {lab: sum(r["label"] == lab for r in chosen) for lab in LABELS}
    print(f"Wrote {len(chosen)} claims to {a.out_file}: {counts}")


if __name__ == "__main__":
    main()
