"""Command line:  python -m truthcheck.cli --demo
                 python -m truthcheck.cli --provider groq --model <name> --wikipedia "text to check"
"""
from __future__ import annotations

import argparse
import json
import sys

from .demo import DEMO_ANSWER, DEMO_DOCS, DEMO_QUESTION
from .llm import make_llm
from .pipeline import TruthCheck
from .retrieval import LocalDocsRetriever, WikipediaRetriever

ICON = {"supported": "[OK]  ", "contradicted": "[WRONG]", "unverified": "[?]   "}


def format_report(rep) -> str:
    lines = []
    if rep.trust_score is not None:
        c = rep.counts()
        lines.append(f"Trust score: {rep.trust_score:.0%}  "
                     f"({c['supported']} supported, {c['contradicted']} contradicted, "
                     f"{c['unverified']} unverified)\n")
    for cr in rep.claims:
        lines.append(f"{ICON[cr.verdict]} {cr.claim}")
        if cr.reasoning:
            lines.append(f"          why: {cr.reasoning}")
        if cr.consistency is not None:
            lines.append(f"          instability when re-asked: {cr.consistency:.0%}")
        if cr.source_numbers:
            lines.append(f"          sources: {cr.source_numbers}")
    if rep.skipped:
        lines.append("\nNot checked (opinion/advice): " + " | ".join(rep.skipped))
    if rep.corrected_answer and rep.corrected_answer != rep.answer:
        lines.append("\nCorrected answer:\n" + rep.corrected_answer)
    if rep.sources:
        lines.append("\nSources:")
        lines += [f"  [{s['n']}] {s['source']} {s['url'] or ''}" for s in rep.sources]
    lines += [f"\nNOTE: {n}" for n in rep.notes]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fact-check an AI answer against evidence.")
    ap.add_argument("text", nargs="?", help="the answer to check (or the question with --ask)")
    ap.add_argument("--file", help="read the text from a file")
    ap.add_argument("--question", help="the question the answer responds to")
    ap.add_argument("--ask", action="store_true", help="treat text as a question; LLM answers, then we check")
    ap.add_argument("--demo", action="store_true", help="offline scripted demo, no API key needed")
    ap.add_argument("--provider"); ap.add_argument("--model"); ap.add_argument("--base-url")
    ap.add_argument("--docs", help="folder of .txt/.md files to use as the knowledge source")
    ap.add_argument("--wikipedia", action="store_true", help="use Wikipedia (default if no --docs)")
    ap.add_argument("--samples", type=int, default=0, help="re-ask the question N times (consistency signal)")
    ap.add_argument("--keep-unverified", action="store_true", help="flag instead of removing unverified claims")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    a = ap.parse_args(argv)

    if a.demo:
        llm, retr = make_llm("demo"), LocalDocsRetriever(DEMO_DOCS)
        text, a.question = a.text or DEMO_ANSWER, a.question or DEMO_QUESTION
    else:
        try:
            llm = make_llm(a.provider, a.model, base_url=a.base_url)
        except ValueError as e:
            sys.exit(f"Setup problem: {e}")
        retr = LocalDocsRetriever.from_folder(a.docs) if a.docs else WikipediaRetriever()
        text = open(a.file, encoding="utf-8").read() if a.file else a.text
    if not text:
        sys.exit("Give some text, --file, or --demo.")

    tc = TruthCheck(llm, retr, consistency_samples=a.samples,
                    unverified_policy="flag" if a.keep_unverified else "remove")
    rep = tc.ask_and_check(text) if a.ask else tc.check(text, a.question)
    print(json.dumps(rep.to_dict(), indent=2) if a.json else format_report(rep))


if __name__ == "__main__":
    main()
