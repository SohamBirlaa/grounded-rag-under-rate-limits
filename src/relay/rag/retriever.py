"""Retrieval engine for The Relay research paper corpus.

Implements the 4 benchmark retrieval matrix settings:
- S1_256_3_no: BM25 on 256-tok chunks, top 3, no reranker
- S2_450_5_no: Dense BGE-small on 450-tok chunks, top 5, no reranker
- S3_450_8to4_rrf_no: 4 BM25 + 4 Dense candidates on 450-tok chunks -> RRF (k=60) -> deduplicate -> top 4, no reranker
- S_opt_structure: Optional structure-aware hybrid arm with structural breadcrumbs

Supports Quota-Conserving Compact Mode (k=2, max 200 tokens/piece) when TPM budget is degraded.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from relay.contracts.schemas import PaperIdType, RetrievalSettingType
from relay.rag.chunker import Chunk, count_tokens
from relay.rag.index import BM25Index, DenseIndex, load_index_cache


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with score, rank, provenance, and retrieval method."""

    chunk_id: str
    paper_id: PaperIdType
    version: str
    page: int
    section: str
    content_type: str
    text: str
    token_count: int
    score: float
    rank: int
    retrieval_method: str  # "bm25", "dense", "hybrid_rrf", "structure_hybrid"
    structural_breadcrumb: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Retriever:
    """Multi-setting in-memory retriever for The Relay."""

    def __init__(
        self,
        bm25_256: Optional[BM25Index] = None,
        bm25_450: Optional[BM25Index] = None,
        dense_450: Optional[DenseIndex] = None,
        bm25_450_struct: Optional[BM25Index] = None,
        dense_450_struct: Optional[DenseIndex] = None,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_256 = bm25_256
        self.bm25_450 = bm25_450
        self.dense_450 = dense_450
        self.bm25_450_struct = bm25_450_struct
        self.dense_450_struct = dense_450_struct
        self.rrf_k = rrf_k

    @classmethod
    def from_cache(cls, cache_dir: Path, rrf_k: int = 60) -> "Retriever":
        """Instantiate Retriever from saved disk cache in data/corpus/indexes/."""
        cache_path = Path(cache_dir)

        # 1. S1: 256-token chunks
        chunks_256, _ = load_index_cache(cache_path, "s1_256")
        bm25_256 = BM25Index(chunks_256, use_breadcrumb=False)

        # 2. S2/S3: 450-token chunks
        chunks_450, embs_450 = load_index_cache(cache_path, "s23_450")
        bm25_450 = BM25Index(chunks_450, use_breadcrumb=False)
        dense_450 = DenseIndex(chunks_450, embeddings=embs_450, use_breadcrumb=False)

        # 3. S_opt_structure: 450-token chunks with breadcrumbs (shares same underlying fastembed model)
        bm25_450_struct = BM25Index(chunks_450, use_breadcrumb=True)
        dense_450_struct = DenseIndex(
            chunks_450,
            embeddings=embs_450,
            use_breadcrumb=True,
            model=dense_450._get_model(),
        )

        return cls(
            bm25_256=bm25_256,
            bm25_450=bm25_450,
            dense_450=dense_450,
            bm25_450_struct=bm25_450_struct,
            dense_450_struct=dense_450_struct,
            rrf_k=rrf_k,
        )

    def warmup(self) -> None:
        """Pre-warm model sessions and caches to eliminate cold-start jitter in benchmarks."""
        if self.dense_450:
            self.dense_450._get_model()
            # Execute dummy warm-up search across settings
            self.retrieve_passages("warmup query transformer attention", setting="S1_256_3_no")
            self.retrieve_passages("warmup query transformer attention", setting="S2_450_5_no")
            self.retrieve_passages("warmup query transformer attention", setting="S3_450_8to4_rrf_no")
            self.retrieve_passages("warmup query transformer attention", setting="S_opt_structure")

    def retrieve_passages(
        self,
        query: str,
        setting: RetrievalSettingType = "S3_450_8to4_rrf_no",
        k: Optional[int] = None,
        compact_mode: bool = False,
    ) -> List[RetrievedChunk]:
        """Execute retrieval according to specified setting and degradation state."""
        if setting == "S1_256_3_no":
            limit = k if k is not None else (2 if compact_mode else 3)
            return self._retrieve_s1(query, top_k=limit, compact_mode=compact_mode)
        elif setting == "S2_450_5_no":
            limit = k if k is not None else (2 if compact_mode else 5)
            return self._retrieve_s2(query, top_k=limit, compact_mode=compact_mode)
        elif setting == "S3_450_8to4_rrf_no":
            limit = k if k is not None else (2 if compact_mode else 4)
            return self._retrieve_s3(query, top_k=limit, compact_mode=compact_mode)
        elif setting == "S_opt_structure":
            limit = k if k is not None else (2 if compact_mode else 5)
            return self._retrieve_s_opt(query, top_k=limit, compact_mode=compact_mode)
        else:
            raise ValueError(f"Unsupported retrieval setting: {setting}")

    def _retrieve_s1(self, query: str, top_k: int, compact_mode: bool) -> List[RetrievedChunk]:
        """S1: BM25 on 256-token chunks."""
        if not self.bm25_256:
            raise RuntimeError("BM25 256-token index is not initialized")

        raw_results = self.bm25_256.search(query, top_k=top_k)
        return self._format_results(raw_results, "bm25", compact_mode)

    def _retrieve_s2(self, query: str, top_k: int, compact_mode: bool) -> List[RetrievedChunk]:
        """S2: Dense BGE-small on 450-token chunks."""
        if not self.dense_450:
            raise RuntimeError("Dense 450-token index is not initialized")

        raw_results = self.dense_450.search(query, top_k=top_k)
        return self._format_results(raw_results, "dense", compact_mode)

    def _retrieve_s3(self, query: str, top_k: int, compact_mode: bool) -> List[RetrievedChunk]:
        """S3: 4 BM25 + 4 Dense candidates on 450-token chunks -> RRF -> top 4."""
        if not self.bm25_450 or not self.dense_450:
            raise RuntimeError("Indexes for S3 (BM25 + Dense 450) are not initialized")

        # Fetch 4 candidates from BM25 and 4 candidates from Dense
        bm25_candidates = self.bm25_450.search(query, top_k=4)
        dense_candidates = self.dense_450.search(query, top_k=4)

        # RRF formula: RRF_Score(d) = sum(1 / (k_rrf + rank_m(d)))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        for chunk, _, rank in bm25_candidates:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for chunk, _, rank in dense_candidates:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort descending by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        selected_cids = sorted_cids[:top_k]

        results: List[RetrievedChunk] = []
        for rank, cid in enumerate(selected_cids, start=1):
            chunk = chunk_map[cid]
            text = self._apply_compact_mode(chunk.text) if compact_mode else chunk.text
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    version=chunk.version,
                    page=chunk.page,
                    section=chunk.section,
                    content_type=chunk.content_type,
                    text=text,
                    token_count=count_tokens(text),
                    score=float(rrf_scores[cid]),
                    rank=rank,
                    retrieval_method="hybrid_rrf",
                    structural_breadcrumb=chunk.structural_breadcrumb,
                    metadata=dict(chunk.metadata),
                )
            )

        return results

    def _retrieve_s_opt(self, query: str, top_k: int, compact_mode: bool) -> List[RetrievedChunk]:
        """S_opt_structure: Structure-breadcrumb hybrid arm."""
        if not self.bm25_450_struct or not self.dense_450_struct:
            raise RuntimeError("Indexes for S_opt_structure are not initialized")

        bm25_candidates = self.bm25_450_struct.search(query, top_k=5)
        dense_candidates = self.dense_450_struct.search(query, top_k=5)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        for chunk, _, rank in bm25_candidates:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for chunk, _, rank in dense_candidates:
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        selected_cids = sorted_cids[:top_k]

        results: List[RetrievedChunk] = []
        for rank, cid in enumerate(selected_cids, start=1):
            chunk = chunk_map[cid]
            text = self._apply_compact_mode(chunk.text) if compact_mode else chunk.text
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    version=chunk.version,
                    page=chunk.page,
                    section=chunk.section,
                    content_type=chunk.content_type,
                    text=text,
                    token_count=count_tokens(text),
                    score=float(rrf_scores[cid]),
                    rank=rank,
                    retrieval_method="structure_hybrid",
                    structural_breadcrumb=chunk.structural_breadcrumb,
                    metadata=dict(chunk.metadata),
                )
            )

        return results

    def _format_results(
        self,
        raw_results: List[Tuple[Chunk, float, int]],
        method: str,
        compact_mode: bool,
    ) -> List[RetrievedChunk]:
        """Convert raw search results to typed RetrievedChunk objects."""
        formatted: List[RetrievedChunk] = []
        for chunk, score, rank in raw_results:
            text = self._apply_compact_mode(chunk.text) if compact_mode else chunk.text
            formatted.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    version=chunk.version,
                    page=chunk.page,
                    section=chunk.section,
                    content_type=chunk.content_type,
                    text=text,
                    token_count=count_tokens(text),
                    score=score,
                    rank=rank,
                    retrieval_method=method,
                    structural_breadcrumb=chunk.structural_breadcrumb,
                    metadata=dict(chunk.metadata),
                )
            )
        return formatted

    @staticmethod
    def _apply_compact_mode(text: str, max_tokens: int = 200) -> str:
        """Truncate chunk text to quota-conserving max token boundary."""
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens]) + " ... [truncated for compact quota mode]"

