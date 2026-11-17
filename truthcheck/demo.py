"""Offline demo. NOT real fact-checking.

DemoLLM is a script that only understands the one sample answer below. It exists so
you can see the whole pipeline (real retrieval + real report building) without an
API key. For anything else, connect a real model with make_llm().
"""
from __future__ import annotations

import json
import re

DEMO_QUESTION = "Tell me about Marie Curie."
DEMO_ANSWER = ("Marie Curie was born in Warsaw. She won two Nobel Prizes. "
               "She discovered penicillin.")

DEMO_DOCS = {
    "Marie Curie": (
        "Marie Curie was born Maria Sklodowska in Warsaw, in what was then the Russian Empire, "
        "in 1867. She won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry "
        "in 1911. She is known for her research on radioactivity."),
    "Penicillin": (
        "Penicillin was discovered in 1928 by the Scottish scientist Alexander Fleming at "
        "St Mary's Hospital in London. It became the first widely used antibiotic."),
    "Eiffel Tower": (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was completed in 1889 "
        "for the World's Fair."),
}

# claim -> (keyword that must appear in the evidence, verdict, correction)
_SCRIPT = {
    "Marie Curie was born in Warsaw.": ("Warsaw", "supported", None),
    "Marie Curie won two Nobel Prizes.": ("Chemistry", "supported", None),
    "Marie Curie discovered penicillin.": (
        "Fleming", "contradicted",
        "Penicillin was discovered by Alexander Fleming in 1928, not by Marie Curie."),
}


class DemoLLM:
    def complete(self, system, user, temperature=0.0, max_tokens=1024):
        if "[task:extract_claims]" in system:
            if DEMO_ANSWER not in user:
                raise ValueError("Demo mode only understands the built-in sample answer. "
                                 "Choose a real provider to check other text.")
            sents = [s.strip() for s in re.split(r"(?<=\.)\s+", DEMO_ANSWER)]
            return json.dumps({"claims": [{"claim": c, "sentence": s}
                                          for c, s in zip(_SCRIPT, sents)], "skipped": []})
        if "[task:verify_claim]" in system:
            claim = re.search(r"CLAIM: (.*)", user).group(1).strip()
            word, verdict, fix = _SCRIPT[claim]
            ids = [int(m.group(1)) for m in re.finditer(r"^\[(\d+)\] .*" + word, user, re.M)]
            if not ids:
                return json.dumps({"verdict": "unverified", "reasoning": "No passage covers this.",
                                   "evidence_ids": [], "corrected_claim": None, "next_query": None})
            why = "The passage states this." if verdict == "supported" else \
                "The passage names someone else for this discovery."
            return json.dumps({"verdict": verdict, "reasoning": why, "evidence_ids": ids[:1],
                               "corrected_claim": fix, "next_query": None})
        if "[task:consistency]" in system:
            return json.dumps({"answers": ["yes", "yes", "no"]})
        if "[task:answer]" in system:
            return DEMO_ANSWER
        if "[task:correct_answer]" in system:
            return ("Marie Curie was born in Warsaw. She won two Nobel Prizes. "
                    "Penicillin was discovered by Alexander Fleming in 1928, not by Marie Curie.")
        raise ValueError("Unknown task for DemoLLM")
