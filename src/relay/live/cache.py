"""Exact and semantic answer caches.

Only answers that passed the sentence gate are cached; refusals caused by
rate limits are never cached.

Semantic cache risk: "d_model of the *base* Transformer" and "... of the *big*
Transformer" embed almost identically but have different answers. A cosine
threshold alone would serve a wrong answer, so a hit additionally requires:
  * the same 'discriminators' (numbers, size words, language pairs, negations,
    table/figure refs, paper routing), and
  * the freshly retrieved evidence to overlap the cached answer's evidence
    (retrieval runs locally and costs no quota, so this check is cheap).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

_DISCRIM_WORDS = {
    "base", "big", "small", "large", "tiny", "medium", "encoder", "decoder", "english", "german", "french",
    "en-de", "en-fr", "not", "without", "except", "only", "first", "last", "train", "training", "test",
    "inference", "minimum", "maximum", "min", "max", "per", "total", "stage", "1-stage", "2-stage",
}


def normalize_query(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s\-\.]", " ", q)
    return " ".join(q.split())


def discriminators(q: str) -> Set[str]:
    nq = normalize_query(q)
    toks = set(nq.split())
    d = {t for t in toks if t in _DISCRIM_WORDS}
    d |= set(re.findall(r"\d+(?:\.\d+)?", nq))
    d |= {m for m in re.findall(r"(table|figure|fig|section|sec)\s*\d+", nq)}
    return d


def cache_key(question: str, setting: str, model: str, corpus_version: str) -> str:
    raw = json.dumps([normalize_query(question), setting, model, corpus_version])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


@dataclass
class CachedAnswer:
    question: str
    answer: str
    evidence_ids: List[str]
    cited_ids: List[str]
    model: str
    setting: str
    created: float = field(default_factory=time.time)
    hits: int = 0


class ExactCache:
    def __init__(self, ttl_s: float = 6 * 3600):
        self.ttl_s = ttl_s
        self._d: Dict[str, CachedAnswer] = {}

    def get(self, key: str) -> Optional[CachedAnswer]:
        e = self._d.get(key)
        if e and time.time() - e.created <= self.ttl_s:
            e.hits += 1
            return e
        return None

    def put(self, key: str, value: CachedAnswer) -> None:
        self._d[key] = value


class SemanticCache:
    def __init__(self, threshold: float = 0.92, min_evidence_overlap: float = 0.5, max_entries: int = 2000,
                 path: Optional[Path] = None):
        self.threshold = threshold
        self.min_evidence_overlap = min_evidence_overlap
        self.max_entries = max_entries
        self.path = path
        self._vecs: List[Any] = []
        self._items: List[CachedAnswer] = []
        self._disc: List[Set[str]] = []
        self.stats = {"lookups": 0, "hits": 0, "blocked_by_discriminator": 0, "blocked_by_evidence": 0}

    def lookup(self, question: str, qvec: Any, evidence_ids: Sequence[str], setting: str, model: str) -> Optional[Dict[str, Any]]:
        import numpy as np

        self.stats["lookups"] += 1
        if not self._vecs:
            return None
        sims = np.asarray(self._vecs) @ np.asarray(qvec)
        d_new = discriminators(question)
        ev_new = set(evidence_ids)
        for i in np.argsort(sims)[::-1][:5]:
            sim = float(sims[i])
            if sim < self.threshold:
                break
            item = self._items[i]
            if item.setting != setting or item.model != model:
                continue
            if self._disc[i] != d_new:
                self.stats["blocked_by_discriminator"] += 1
                continue
            need = set(item.cited_ids) or set(item.evidence_ids)
            overlap = len(need & ev_new) / max(1, len(need))
            if overlap < self.min_evidence_overlap:
                self.stats["blocked_by_evidence"] += 1
                continue
            item.hits += 1
            self.stats["hits"] += 1
            return {"item": item, "similarity": sim, "evidence_overlap": overlap}
        return None

    def add(self, question: str, qvec: Any, item: CachedAnswer) -> None:
        if len(self._items) >= self.max_entries:
            self._vecs.pop(0); self._items.pop(0); self._disc.pop(0)
        self._vecs.append(qvec)
        self._items.append(item)
        self._disc.append(discriminators(question))
