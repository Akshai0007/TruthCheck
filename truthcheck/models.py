"""Plain data containers shared by every part of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

SUPPORTED, CONTRADICTED, UNVERIFIED = "supported", "contradicted", "unverified"


@dataclass
class Evidence:
    id: int
    text: str
    source: str
    url: Optional[str] = None


@dataclass
class ClaimResult:
    claim: str
    sentence: str = ""
    verdict: str = UNVERIFIED
    reasoning: str = ""
    corrected_claim: Optional[str] = None
    evidence: list = field(default_factory=list)        # everything the judge saw
    cited_ids: list = field(default_factory=list)       # what the judge relied on
    searches: list = field(default_factory=list)        # queries tried
    consistency: Optional[float] = None                 # 0 = stable ... 1 = unstable
    source_numbers: list = field(default_factory=list)  # numbers in Report.sources


@dataclass
class Report:
    answer: str
    question: Optional[str] = None
    claims: list = field(default_factory=list)
    skipped: list = field(default_factory=list)         # opinions / advice we did not check
    sources: list = field(default_factory=list)         # [{"n", "source", "url", "text"}]
    trust_score: Optional[float] = None                 # supported / checked claims
    corrected_answer: Optional[str] = None
    notes: list = field(default_factory=list)           # warnings, errors, truncations

    def counts(self) -> dict:
        out = {SUPPORTED: 0, CONTRADICTED: 0, UNVERIFIED: 0}
        for c in self.claims:
            out[c.verdict] = out.get(c.verdict, 0) + 1
        return out

    def to_dict(self) -> dict:
        return asdict(self)
