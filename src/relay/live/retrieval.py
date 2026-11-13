"""Retrieval optimisation for the live path.

Measured problems in the existing index (data/corpus/indexes/s23_450_chunks.json):
  * 64/296 chunks have a section name that is really chart/table numbers
    ("0 10 20 30 40 50"), so citations point at nonsense sections.
  * 34 chunks are reference lists: they match query words (EnCodec, Transformer)
    but contain no answers -> classic naive-retriever trap.
  * 29 chunks exceed 600 tokens (max 3,290) -> blow the 6K TPM budget.

This module (1) repairs section names from the previous valid heading,
(2) drops reference/axis-noise chunks from candidates, (3) routes queries that
name a paper to that paper, (4) fuses BM25 + dense with RRF over a wider pool,
(5) optionally re-ranks with a cross-encoder, and (6) fans out one retrieval per
paper in parallel for cross-paper questions. Every setting is named so it can be
compared in Table 3.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

PAPER_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "1706.03762v7": ("attention is all you need", "vaswani", "wmt 2014", "transformer (big)",
                     "transformer base", "base transformer", "big transformer", "scaled dot-product",
                     "sinusoidal", "english-to-german", "english-to-french", "en-de", "en-fr"),
    "2410.00037v1": ("moshi", "mimi", "helium", "inner monologue", "depth transformer",
                     "temporal transformer", "full-duplex", "spoken dialogue", "audio codec"),
    "2507.07955v2": ("dynamic chunking", "h-net", "hnet", "hierarchical sequence", "chunking",
                     "byte-level", "raw byte", "routing module", "dechunk"),
}

_NUMERIC_SECTION = re.compile(r"^[\d\.\s\-–,%()]+$")
_REFERENCE_SECTION = re.compile(r"^\s*(references|bibliography)\s*$", re.I)
_NUMBERED_HEADING = re.compile(r"^[1-9]\d?(\.[1-9]\d?){0,3}\.?\s+[A-Z][A-Za-z\-]{2,}")


def expand_query(query: str) -> str:
    """BM25 tokenizes with \\b[a-z0-9]+\\b, so "d_model" yields NO terms at all. Add "d model" and the
    glued form "dmodel" (which is how the PDF text spells it)."""
    extra = []
    for ident in re.findall(r"\b\w*_\w+\b", query):
        extra += [ident.replace("_", " "), ident.replace("_", "")]
    return query + (" " + " ".join(extra) if extra else "")


def is_numeric_noise(c: Any) -> bool:
    """Prose chunks that are really figure axes / table cells without headers (letters < 20% of tokens)."""
    if getattr(c, "content_type", "prose") == "table":
        return False  # real tables keep their header row and can answer numeric questions
    toks = (c.text or "").split()
    return bool(toks) and sum(bool(re.search(r"[A-Za-z]{3,}", w)) for w in toks) / len(toks) < 0.2


def detect_papers(query: str) -> List[str]:
    q = query.lower()
    return [pid for pid, kws in PAPER_KEYWORDS.items() if any(k in q for k in kws)]


def is_junk_section(section: str) -> bool:
    """True when the parser used chart/table numbers or citation page lists as a heading."""
    s = (section or "").strip()
    if not s or _NUMERIC_SECTION.match(s):
        return True
    if s[0].isdigit() and not _NUMBERED_HEADING.match(s):
        return True  # "0.75 H-Net (2-stage) overtakes", "8 4.88 25.5 80", "10 - 16, 2023, ..."
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(1, len(s)) > 0.35


def repair_sections(chunks: Sequence[Any]) -> List[Any]:
    """Return copies of chunks with junk section names replaced by the last valid heading in reading order."""
    def order_key(c: Any) -> Tuple[str, int, str]:
        return (c.paper_id, int(c.page), c.chunk_id)

    out: List[Any] = []
    last_valid: Dict[str, str] = {}
    for c in sorted(chunks, key=order_key):
        sec = c.section
        if is_junk_section(sec):
            fixed = last_valid.get(c.paper_id, "unknown section")
            md = dict(getattr(c, "metadata", {}) or {})
            md["section_repaired_from"] = sec
            c2 = replace(c, section=fixed, metadata=md)
        else:
            last_valid[c.paper_id] = sec
            c2 = c
        out.append(c2)
    return out


def is_reference_chunk(c: Any) -> bool:
    if _REFERENCE_SECTION.match(c.section or "") or "_References_" in c.chunk_id:
        return True
    text = c.text or ""
    # Dense bibliography: many "[12]" / "et al." / "arXiv" hits per 100 words
    words = max(1, len(text.split()))
    hits = len(re.findall(r"\bet al\.|arXiv|\bIn Proceedings\b|\b(19|20)\d\d\b", text))
    return hits / words > 0.08


@dataclass
class Candidate:
    chunk: Any
    rrf: float = 0.0
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    via_paper: Optional[str] = None


@dataclass
class RetrievalResult:
    setting: str
    chunks: List[Any]
    scores: List[float]
    routed_papers: List[str]
    subqueries: List[str]
    top_rerank_score: Optional[float]
    ms: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalSetting:
    name: str            # label written to results.csv
    pool: int = 10       # candidates per index
    top_k: int = 4
    rerank: bool = False
    route_papers: bool = True
    filter_junk: bool = True
    compress: bool = True
    fanout: bool = True


# Named settings for Table 3 (existing S1/S3 are run through the legacy Retriever).
SETTINGS: Dict[str, RetrievalSetting] = {
    "R4_hybrid_route": RetrievalSetting("450 tok, 10+10->4 RRF, routed, re-rank no", rerank=False),
    "R5_hybrid_route_rerank": RetrievalSetting("450 tok, 10+10->4 RRF, routed, re-rank yes", rerank=True),
    "R6_rerank_nocompress": RetrievalSetting("450 tok, 10+10->4 RRF, routed, re-rank yes, no compression",
                                             rerank=True, compress=False),
    "R7_compact": RetrievalSetting("450 tok, 10+10->2 RRF, routed, re-rank yes (compact)", top_k=2, rerank=True),
}


class LiveRetriever:
    """Wraps BM25 + dense indexes built over the 450-token chunks."""

    def __init__(self, chunks: Sequence[Any], bm25: Any, dense: Any, reranker: Any = None, rrf_k: int = 60):
        self.chunks = list(chunks)
        self.bm25 = bm25
        self.dense = dense
        self._reranker = reranker
        self.rrf_k = rrf_k
        self.by_id = {c.chunk_id: c for c in self.chunks}

    @classmethod
    def from_index_cache(cls, cache_dir: str = "data/corpus/indexes") -> "LiveRetriever":
        from pathlib import Path

        from relay.rag.index import BM25Index, DenseIndex, load_index_cache

        raw, embs = load_index_cache(Path(cache_dir), "s23_450")
        # keep embedding rows aligned with chunk order
        emb_by_id = {c.chunk_id: embs[i] for i, c in enumerate(raw)} if embs is not None else {}
        fixed = repair_sections(raw)
        import numpy as np

        mat = np.array([emb_by_id[c.chunk_id] for c in fixed], dtype=np.float32) if emb_by_id else None
        bm25 = BM25Index(fixed, use_breadcrumb=False)
        dense = DenseIndex(fixed, embeddings=mat, use_breadcrumb=False)
        return cls(fixed, bm25, dense)

    # ---------------- embeddings shared with the semantic cache ----------------
    def embed_query(self, text: str):
        import numpy as np

        model = self.dense._get_model()
        v = np.asarray(list(model.embed([text]))[0], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def _get_reranker(self):
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        return self._reranker

    def warmup(self, rerank: bool = True) -> None:
        self.embed_query("warmup")
        self.bm25.search("warmup attention", top_k=1)
        if rerank:
            list(self._get_reranker().rerank("warmup", ["warmup passage"]))

    # ---------------- core search ----------------
    def _search_one(self, query: str, setting: RetrievalSetting, paper: Optional[str]) -> List[Candidate]:
        wide = setting.pool * (4 if paper else 1)  # widen pool when we will filter by paper
        cands: Dict[str, Candidate] = {}
        for method, results in (("bm25", self.bm25.search(expand_query(query), top_k=wide)),
                                ("dense", self.dense.search(query, top_k=wide))):
            kept = 0
            for chunk, _score, _rank in results:
                if paper and chunk.paper_id != paper:
                    continue
                if setting.filter_junk and (is_reference_chunk(chunk) or is_numeric_noise(chunk)):
                    continue
                kept += 1
                if kept > setting.pool:
                    break
                c = cands.setdefault(chunk.chunk_id, Candidate(chunk=chunk, via_paper=paper))
                c.rrf += 1.0 / (self.rrf_k + kept)
                if method == "bm25":
                    c.bm25_rank = kept
                else:
                    c.dense_rank = kept
        return sorted(cands.values(), key=lambda c: c.rrf, reverse=True)

    def _rerank(self, query: str, cands: List[Candidate]) -> List[Candidate]:
        if not cands:
            return cands
        # Measured: re-ranking 12 x 2000-char passages added ~0.8-1.2 s TTFT on CPU. Cap length and count.
        scores = list(self._get_reranker().rerank(query, [c.chunk.text[:1200] for c in cands]))
        for c, s in zip(cands, scores):
            c.rerank_score = float(s)
        return sorted(cands, key=lambda c: c.rerank_score, reverse=True)

    def search(self, query: str, setting: RetrievalSetting, paper: Optional[str] = None) -> List[Candidate]:
        cands = self._search_one(query, setting, paper)
        cands = cands[: max(setting.top_k * 2, 8)]
        if setting.rerank:
            cands = self._rerank(query, cands)
        return cands

    async def retrieve(self, query: str, setting: RetrievalSetting) -> RetrievalResult:
        """The pipeline's `retrieve_passages` tool. Runs per-paper searches in parallel threads."""
        import time

        t0 = time.perf_counter()
        papers = detect_papers(query) if setting.route_papers else []
        if setting.fanout and len(papers) >= 2:
            # Cross-paper question: one search per named paper, concurrently, then interleave.
            per = await asyncio.gather(*[asyncio.to_thread(self.search, query, setting, p) for p in papers])
            take = max(1, setting.top_k // len(papers))
            merged: List[Candidate] = []
            for lst in per:
                merged.extend(lst[:take])
            for lst in per:  # fill remaining slots by score
                for c in lst[take:]:
                    if len(merged) >= setting.top_k:
                        break
                    if c not in merged:
                        merged.append(c)
            chosen = merged[: setting.top_k]
            subqueries = [f"{query} [paper={p}]" for p in papers]
        else:
            paper = papers[0] if len(papers) == 1 else None
            chosen = (await asyncio.to_thread(self.search, query, setting, paper))[: setting.top_k]
            subqueries = [query]
        rr = [c.rerank_score for c in chosen if c.rerank_score is not None]
        return RetrievalResult(
            setting=setting.name,
            chunks=[c.chunk for c in chosen],
            scores=[c.rerank_score if c.rerank_score is not None else c.rrf for c in chosen],
            routed_papers=papers,
            subqueries=subqueries,
            top_rerank_score=max(rr) if rr else None,
            ms=(time.perf_counter() - t0) * 1000.0,
            debug={"candidates": [(c.chunk.chunk_id, c.bm25_rank, c.dense_rank, c.rerank_score) for c in chosen]},
        )


class LegacySettingAdapter:
    """Runs the original S1/S3 settings through the existing Retriever, for an apples-to-apples Table 3."""

    LABELS = {
        "S1_256_3_no": "256 tok, 3 pieces, re-rank no",
        "S3_450_8to4_rrf_no": "450 tok, 8->4 via RRF, re-rank no",
    }

    def __init__(self, legacy_retriever: Any, live: LiveRetriever):
        self.legacy = legacy_retriever
        self.live = live

    async def retrieve(self, query: str, setting_key: str) -> RetrievalResult:
        import time

        t0 = time.perf_counter()
        res = await asyncio.to_thread(self.legacy.retrieve_passages, query, setting_key)
        # map to repaired chunk objects (same chunk_id) so citations use repaired sections where possible
        chunks = [self.live.by_id.get(r.chunk_id, r) for r in res]
        return RetrievalResult(setting=self.LABELS[setting_key], chunks=chunks, scores=[r.score for r in res],
                               routed_papers=[], subqueries=[query], top_rerank_score=None,
                               ms=(time.perf_counter() - t0) * 1000.0)
