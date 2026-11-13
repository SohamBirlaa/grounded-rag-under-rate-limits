"""Prompt size reduction: extractive context compression.

Steps per retrieved chunk: (1) drop near-duplicate chunks (overlapping windows),
(2) split into sentences / table rows, (3) score each by query-term overlap
(IDF-weighted) plus a bonus for numbers when the question asks for a value,
(4) keep the best units *and their neighbours* in original order until the
per-chunk and global token budgets are met. Table header rows are always kept.

Risk (measured in evals as `fact_retention`): compression can cut the sentence
that holds the answer. That is why it is a separate Table 3 setting.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_WORD = re.compile(r"[A-Za-z][A-Za-z\-]+|\d+(?:\.\d+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
STOP = set("""a an the of in on for to and or is are was were be been by with as at from that this these those
what which who whom how does do did can could would should into than then their its it they them there here
about over under per via using use used paper papers model models according""".split())
NUMERIC_Q = re.compile(r"\b(what|how)\s+(is|are|was|were|many|much)\b|\bvalue|\bscore|\bbleu|\bnumber|\brate\b|\bsize|"
                       r"\bdimension|\bcount|\bhow long|\blatency|\bhz\b|\bparameters?\b", re.I)


def tok_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def terms(text: str) -> List[str]:
    out = []
    for w in _WORD.findall((text or "").lower()):
        if w in STOP or len(w) < 2:
            continue
        out.append(w[:-1] if w.endswith("s") and len(w) > 4 else w)  # crude plural strip
    return out


def shingles(text: str, n: int = 5) -> Set[str]:
    w = (text or "").lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def split_units(text: str, is_table: bool) -> List[str]:
    if is_table:
        return [ln for ln in (text or "").splitlines() if ln.strip()]
    units: List[str] = []
    for para in re.split(r"\n{2,}", text or ""):
        para = " ".join(para.split())
        if para:
            units.extend(s for s in _SENT_SPLIT.split(para) if s.strip())
    return units


@dataclass
class Evidence:
    tag: str            # "E1"
    chunk: Any          # original chunk (paper_id, page, section, chunk_id)
    text: str           # text placed in the prompt
    tokens_before: int
    tokens_after: int


class Compressor:
    def __init__(self, corpus_texts: Iterable[str]):
        df: Dict[str, int] = {}
        n = 0
        for t in corpus_texts:
            n += 1
            for w in set(terms(t)):
                df[w] = df.get(w, 0) + 1
        self.n_docs = max(1, n)
        self.df = df

    def idf(self, w: str) -> float:
        return math.log(1 + self.n_docs / (1 + self.df.get(w, 0)))

    def _score(self, unit: str, qterms: Set[str], numeric: bool) -> float:
        ut = set(terms(unit))
        s = sum(self.idf(w) for w in ut & qterms)
        if numeric and re.search(r"\d", unit):
            s += 1.0
        return s

    def build(
        self,
        question: str,
        chunks: Sequence[Any],
        compress: bool = True,
        per_chunk_tokens: int = 220,
        total_tokens: int = 900,
        dedupe_threshold: float = 0.5,
    ) -> List[Evidence]:
        qterms = set(terms(question))
        numeric = bool(NUMERIC_Q.search(question))
        kept: List[Evidence] = []
        seen: List[Set[str]] = []
        used = 0
        for c in chunks:
            text = c.text or ""
            sh = shingles(text)
            if any(jaccard(sh, s) >= dedupe_threshold for s in seen):
                continue  # overlapping window of a chunk we already have
            seen.append(sh)
            before = tok_count(text)
            budget = min(per_chunk_tokens, total_tokens - used)
            if budget <= 40:
                break
            if not compress:
                body = text if before <= max(budget, 600) else " ".join(text.split()[: max(budget, 600)])
            else:
                body = self._compress_one(text, getattr(c, "content_type", "prose") == "table", qterms, numeric, budget)
            after = tok_count(body)
            used += after
            kept.append(Evidence(tag=f"E{len(kept) + 1}", chunk=c, text=body, tokens_before=before, tokens_after=after))
        return kept

    def _compress_one(self, text: str, is_table: bool, qterms: Set[str], numeric: bool, budget: int) -> str:
        units = split_units(text, is_table)
        if not units:
            return ""
        if tok_count(text) <= budget:
            return "\n".join(units) if is_table else " ".join(units)
        scores = [self._score(u, qterms, numeric) for u in units]
        order = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)
        chosen: Set[int] = {0} if is_table else set()  # table header row
        used = sum(tok_count(units[i]) for i in chosen)
        for i in order:
            if scores[i] <= 0 and chosen:
                break
            for j in (i, i - 1, i + 1):  # keep neighbours: answers often span two sentences
                if 0 <= j < len(units) and j not in chosen:
                    t = tok_count(units[j])
                    if used + t <= budget:
                        chosen.add(j)
                        used += t
            if used >= budget:
                break
        idx = sorted(chosen)
        sep = "\n" if is_table else " "
        parts, prev = [], None
        for i in idx:
            if prev is not None and i != prev + 1 and not is_table:
                parts.append("…")
            parts.append(units[i])
            prev = i
        return sep.join(parts)
