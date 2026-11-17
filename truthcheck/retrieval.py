"""Finding evidence.

Passage windows (a few sentences each) come from Factcheck-GPT's chunking;
BM25 ranking comes from OpenFActScore. Two sources are provided:
  LocalDocsRetriever  -> your own .txt/.md files (the "RAG" part, e.g. a college FAQ)
  WikipediaRetriever  -> live Wikipedia through its public API
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol
from urllib.parse import quote

import requests

_STOP = set("a an the and or of in on at to is are was were be been it its this that these those "
            "for with as by from has have had he she they his her their which who whom what when "
            "where also not but".split())


@dataclass
class Passage:
    text: str
    source: str
    url: Optional[str] = None
    score: float = 0.0


class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list: ...


def tokenize(text: str) -> list:
    """Lower-cased words in any script. Combining marks (accents, Devanagari vowel
    signs) stay attached to their word instead of splitting it."""
    tokens, cur = [], []
    for ch in text.lower():
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return [t for t in tokens if t not in _STOP]


_ABBREV = {"dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "vs", "etc", "no", "inc", "ltd",
           "co", "mt", "gen", "col", "sgt", "capt", "lt", "rev", "hon", "fig", "approx"}


def split_sentences(text: str) -> list:
    """Split after . ! ? or the Hindi danda, but keep 'Dr. Smith', 'J. K. Rowling' together."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?\u0964])\s+(?![a-z])", text.strip()) if p.strip()]
    out = []
    for part in parts:
        if out:
            last = out[-1].split()[-1].rstrip(".").lower()
            if out[-1].endswith(".") and (last in _ABBREV or (len(last) == 1 and last.isalpha())):
                out[-1] += " " + part
                continue
        out.append(part)
    return out


def text_from_file(name: str, data: bytes) -> str:
    """Read .txt/.md/.pdf bytes. PDF needs the optional `pypdf` package."""
    if name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError(f"Reading {name} needs pypdf:  pip install pypdf")
        import io
        return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


def make_passages(text: str, source: str, url: Optional[str] = None,
                  window: int = 3, stride: int = 2) -> list:
    """Overlapping windows of `window` sentences, paragraph by paragraph."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        sents = split_sentences(" ".join(para.split()))
        if not sents:
            continue
        if len(sents) <= window:
            out.append(Passage(" ".join(sents), source, url))
            continue
        for i in range(0, len(sents), stride):
            out.append(Passage(" ".join(sents[i:i + window]), source, url))
            if i + window >= len(sents):
                break
    return out


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.n if self.n else 0.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query_tokens: list) -> list:
        out = []
        for tf, d in zip(self.tf, self.docs):
            s, dl = 0.0, len(d)
            for t in set(query_tokens):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


def rank(query: str, passages: list, k: int = 4) -> list:
    """Top-k passages by BM25; passages with no word overlap are dropped."""
    if not passages:
        return []
    bm = BM25([tokenize(p.text) for p in passages])
    scores = bm.scores(tokenize(query))
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [Passage(p.text, p.source, p.url, s) for p, s in ranked[:k] if s > 0]


class LocalDocsRetriever:
    def __init__(self, docs: dict):
        """docs: {name: full text}"""
        self.passages = []
        for name, text in docs.items():
            self.passages += make_passages(text, name)

    @classmethod
    def from_folder(cls, folder: str):
        docs = {p.name: text_from_file(p.name, p.read_bytes())
                for p in sorted(Path(folder).rglob("*"))
                if p.suffix.lower() in (".txt", ".md", ".pdf")}
        if not docs:
            raise ValueError(f"No .txt, .md or .pdf files found in {folder}")
        return cls(docs)

    def search(self, query: str, k: int = 4) -> list:
        return rank(query, self.passages, k)


class WikipediaRetriever:
    """Search Wikipedia, download the top pages, keep the best-matching passages."""

    def __init__(self, n_pages: int = 3, lang: str = "en", session=None, timeout: int = 15,
                 api_url: Optional[str] = None):
        self.api = api_url or f"https://{lang}.wikipedia.org/w/api.php"
        self.wiki = f"https://{lang}.wikipedia.org/wiki/"
        self.n_pages, self.timeout = n_pages, timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": "truthcheck-student-project/0.1 (educational fact-checking demo)"})
        self._pages = {}

    def _get(self, params: dict) -> dict:
        r = self.session.get(self.api, params={**params, "format": "json", "formatversion": 2},
                             timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _page_passages(self, title: str) -> list:
        if title not in self._pages:
            data = self._get({"action": "query", "prop": "extracts", "explaintext": 1,
                              "exsectionformat": "plain", "redirects": 1, "titles": title})
            pages = data.get("query", {}).get("pages", [])
            text = (pages[0].get("extract") or "")[:30000] if pages else ""
            url = self.wiki + quote(title.replace(" ", "_"))
            self._pages[title] = make_passages(text, f"Wikipedia: {title}", url)
        return self._pages[title]

    def search(self, query: str, k: int = 4) -> list:
        data = self._get({"action": "query", "list": "search",
                          "srsearch": query, "srlimit": self.n_pages})
        titles = [h["title"] for h in data.get("query", {}).get("search", [])]
        passages = []
        for t in titles:
            passages += self._page_passages(t)
        return rank(query, passages, k)
