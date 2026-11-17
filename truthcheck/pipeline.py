"""The fact-checking agent.

Flow:  answer -> extract claims -> (search -> judge)* per claim -> score -> correct

Ideas used (details in README):
  VeriScore / Factcheck-GPT   check only verifiable claims, standalone wording
  SAFE                        judge may ask for ONE new search query when unsure
  Chain-of-Verification       the judge never sees the original answer
  SelfCheckGPT                optional "ask again" consistency signal
  RARR                        minimal-edit correction
  RAGAS / FActScore           trust score = supported claims / checked claims
  OpenFActScore               skip answers where the model simply abstained
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from typing import Optional

from . import prompts as P
from .models import (CONTRADICTED, SUPPORTED, UNVERIFIED, ClaimResult, Evidence, Report)

_ABSTAIN = re.compile(r"\b(i (do not|don't|cannot|can't|am unable to) (know|answer|help|provide)|"
                      r"i'?m not sure|as an ai)\b", re.I)
_SYNONYMS = {"true": SUPPORTED, "supports": SUPPORTED, "refuted": CONTRADICTED,
             "false": CONTRADICTED, "refutes": CONTRADICTED, "inconclusive": UNVERIFIED,
             "not enough info": UNVERIFIED, "unsupported": UNVERIFIED}


def parse_json(text: str):
    """Safe JSON extraction (Factcheck-GPT used eval(); we never do)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        s, e = t.find(opener), t.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("the model did not return valid JSON")


def normalize_verdict(v) -> str:
    v = str(v or "").lower().strip()
    v = _SYNONYMS.get(v, v)
    return v if v in (SUPPORTED, CONTRADICTED, UNVERIFIED) else UNVERIFIED


class TruthCheck:
    def __init__(self, llm, retriever, max_claims: int = 12, max_search_steps: int = 2,
                 evidence_k: int = 4, workers: int = 3, consistency_samples: int = 0,
                 unverified_policy: str = "remove"):
        if unverified_policy not in ("remove", "flag"):
            raise ValueError("unverified_policy must be 'remove' or 'flag'")
        self.llm, self.retriever = llm, retriever
        self.max_claims, self.max_search_steps = max_claims, max_search_steps
        self.evidence_k, self.workers = evidence_k, workers
        self.consistency_samples, self.unverified_policy = consistency_samples, unverified_policy

    # ---- helpers -------------------------------------------------------
    def _ask_json(self, system: str, user: str, retries: int = 1, list_key=None, **kw) -> dict:
        """Ask for a JSON object; re-ask once if the reply is malformed or the wrong shape."""
        err = None
        for _ in range(retries + 1):
            try:
                data = parse_json(self.llm.complete(system, user, **kw))
                if isinstance(data, list) and list_key:
                    data = {list_key: data}
                if not isinstance(data, dict):
                    raise ValueError("the model returned JSON of the wrong shape (expected an object)")
                return data
            except ValueError as e:
                err = e
        raise ValueError(str(err))

    # ---- step 1: extract claims ---------------------------------------
    def extract_claims(self, answer: str, question: Optional[str] = None):
        user = (f"QUESTION (context only): {question}\n\n" if question else "")
        user += f"TEXT:\n{answer}\n\nMaximum claims: {self.max_claims}"
        data = self._ask_json(P.EXTRACT_SYSTEM, user, list_key="claims", max_tokens=2048)
        claims, seen = [], set()
        for item in data.get("claims") or []:
            text = (item.get("claim") if isinstance(item, dict) else str(item)) or ""
            text = text.strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                sent = (item.get("sentence", "") if isinstance(item, dict) else "") or ""
                claims.append(ClaimResult(claim=text, sentence=sent.strip()))
        return claims, [str(s) for s in (data.get("skipped") or [])]

    # ---- step 2: verify one claim -------------------------------------
    def verify_claim(self, claim: str, sentence: str = "") -> ClaimResult:
        res = ClaimResult(claim=claim, sentence=sentence)
        evidence, seen, query = [], set(), claim
        try:
            for step in range(max(1, self.max_search_steps)):
                res.searches.append(query)
                for p in self.retriever.search(query, self.evidence_k):
                    if p.text not in seen and len(evidence) < 8:
                        seen.add(p.text)
                        evidence.append(Evidence(len(evidence) + 1, p.text, p.source, p.url))
                if not evidence:
                    res.reasoning = "No evidence was found for this claim."
                    break
                shown = "\n".join(f"[{e.id}] ({e.source}) {e.text}" for e in evidence)
                out = self._ask_json(P.VERIFY_SYSTEM, f"CLAIM: {claim}\n\nEVIDENCE:\n{shown}")
                res.verdict = normalize_verdict(out.get("verdict"))
                res.reasoning = str(out.get("reasoning") or "").strip()
                res.cited_ids = [int(i) for i in (out.get("evidence_ids") or [])
                                 if str(i).isdigit() and 1 <= int(i) <= len(evidence)]
                res.corrected_claim = out.get("corrected_claim") or None
                nxt = (out.get("next_query") or "").strip()
                if res.verdict == UNVERIFIED and nxt and nxt != query \
                        and step < self.max_search_steps - 1:
                    query = nxt  # SAFE-style: try one new query for the missing facts
                    continue
                break
        except Exception as e:  # never let one claim crash the whole report
            res.verdict, res.reasoning = UNVERIFIED, f"Checker error: {e}"
        res.evidence = evidence
        # Guard: a verdict with no cited evidence would be the judge's memory talking.
        if res.verdict in (SUPPORTED, CONTRADICTED) and not res.cited_ids:
            res.reasoning += " [Downgraded: the judge cited no evidence.]"
            res.verdict, res.corrected_claim = UNVERIFIED, None
        if res.verdict != CONTRADICTED:
            res.corrected_claim = None
        return res

    # ---- step 3 (optional): ask again, look for instability -----------
    def _consistency(self, question: str, claims: list, notes: list):
        numbered = "\n".join(f"{i + 1}. {c.claim}" for i, c in enumerate(claims))
        scores = [[] for _ in claims]
        for _ in range(self.consistency_samples):
            try:
                sample = self.llm.complete(P.ANSWER_SYSTEM, question, temperature=1.0)
                data = self._ask_json(P.CONSIST_SYSTEM, f"PASSAGE:\n{sample}\n\nCLAIMS:\n{numbered}")
                for i, a in enumerate(data.get("answers", [])[:len(claims)]):
                    scores[i].append({"yes": 0.0, "no": 1.0}.get(str(a).lower().strip(), 0.5))
            except Exception as e:
                notes.append(f"Consistency check skipped a sample: {e}")
        for c, s in zip(claims, scores):
            c.consistency = round(mean(s), 2) if s else None

    # ---- step 4: correct the answer -----------------------------------
    def correct(self, answer: str, claims: list, sources: list) -> str:
        actions = []
        for i, c in enumerate(claims, 1):
            if c.verdict == SUPPORTED:
                act = "keep"
            elif c.verdict == CONTRADICTED and c.corrected_claim:
                act = "fix"
            elif c.verdict == UNVERIFIED and self.unverified_policy == "flag":
                act = "flag"
            else:
                act = "remove"
            row = f"{i}. ACTION={act} | SENTENCE={c.sentence or c.claim} | CLAIM={c.claim}"
            if act == "fix":
                row += f" | REPLACEMENT={c.corrected_claim}"
            if c.source_numbers and act in ("keep", "fix"):
                row += f" | SOURCES={c.source_numbers}"
            actions.append(row)
        text = self.llm.complete(P.CORRECT_SYSTEM,
                                 f"ORIGINAL ANSWER:\n{answer}\n\nCLAIMS:\n" + "\n".join(actions),
                                 max_tokens=2048).strip()
        valid = {s["n"] for s in sources}  # drop any citation number the model invented
        text = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip() or "No statement in this answer could be verified."

    # ---- the whole flow -------------------------------------------------
    def check(self, answer: str, question: Optional[str] = None) -> Report:
        rep = Report(answer=answer, question=question)
        if len(answer) < 300 and _ABSTAIN.search(answer):
            rep.notes.append("The answer is a refusal / 'I don't know', so there is nothing to check.")
            return rep
        try:
            claims, rep.skipped = self.extract_claims(answer, question)
        except Exception as e:
            rep.notes.append(f"Could not extract claims: {e}")
            return rep
        if not claims:
            rep.notes.append("No checkable factual claims were found in this answer.")
            return rep
        if len(claims) > self.max_claims:
            rep.notes.append(f"Only the first {self.max_claims} of {len(claims)} claims were checked.")
            claims = claims[:self.max_claims]

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            rep.claims = list(pool.map(lambda c: self.verify_claim(c.claim, c.sentence), claims))
        errors = sum("Checker error" in c.reasoning for c in rep.claims)
        if errors:
            rep.notes.append(f"{errors} claim(s) could not be checked because of errors (see reasoning).")

        if self.consistency_samples > 0:
            if question:
                self._consistency(question, rep.claims, rep.notes)
            else:
                rep.notes.append("Consistency check skipped: it needs the original question "
                                 "(use 'Ask a question' mode or provide the question).")

        index = {}  # one numbered source list for the whole report
        for c in rep.claims:
            for i in c.cited_ids:
                e = c.evidence[i - 1]
                key = (e.source, e.text)
                if key not in index:
                    index[key] = len(index) + 1
                    rep.sources.append({"n": index[key], "source": e.source, "url": e.url, "text": e.text})
                c.source_numbers.append(index[key])

        rep.trust_score = round(sum(c.verdict == SUPPORTED for c in rep.claims) / len(rep.claims), 2)
        needs_fix = any(c.verdict != SUPPORTED for c in rep.claims)
        try:
            rep.corrected_answer = self.correct(answer, rep.claims, rep.sources) if needs_fix else answer
        except Exception as e:
            rep.notes.append(f"Could not write the corrected answer: {e}")
        return rep

    def ask_and_check(self, question: str) -> Report:
        """Let the LLM answer first, then fact-check what it said."""
        return self.check(self.llm.complete(P.ANSWER_SYSTEM, question), question)
