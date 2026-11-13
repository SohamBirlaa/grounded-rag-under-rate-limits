"""In-memory indexing for The Relay research paper corpus.

Provides:
- BM25Index: Okapi BM25 implementation (k1=1.5, b=0.75) with IDF caching.
- DenseIndex: fastembed BAAI/bge-small-en-v1.5 normalized cosine similarity search.
- Precomputed index persistence (JSON + NPY) for sub-10ms startup and offline retrieval.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from relay.rag.chunker import Chunk

# Pre-compiled regex for fast alphanumeric tokenization
TOKEN_PATTERN = re.compile(r"\b[a-z0-9]+\b")


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 indexing."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 index with Robertson/Lucene IDF formulation."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        use_breadcrumb: bool = False,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.use_breadcrumb = use_breadcrumb
        self.chunks: List[Chunk] = list(chunks)
        self.n_docs = len(self.chunks)

        # Inverted index: term -> list of (doc_idx, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}

        if self.n_docs > 0:
            self._build_index()

    def _build_index(self) -> None:
        """Construct inverted index, calculate doc lengths, and precompute IDF."""
        total_len = 0
        term_doc_counts: Dict[str, int] = {}

        for doc_idx, chunk in enumerate(self.chunks):
            text = chunk.to_retrieval_text(include_breadcrumb=self.use_breadcrumb)
            tokens = tokenize_text(text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            # Term frequencies for this document
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, count in tf_map.items():
                self.inverted_index.setdefault(term, []).append((doc_idx, count))
                term_doc_counts[term] = term_doc_counts.get(term, 0) + 1

        self.avg_doc_len = total_len / max(1, self.n_docs)

        # Precompute Robertson IDF with +1 smoothing: ln((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in term_doc_counts.items():
            numerator = self.n_docs - df + 0.5
            denominator = df + 0.5
            self.idf[term] = math.log((numerator / denominator) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float, int]]:
        """Search BM25 index and return list of (Chunk, score, rank)."""
        if self.n_docs == 0:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        # Accumulate BM25 scores
        scores = np.zeros(self.n_docs, dtype=np.float32)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                doc_len = self.doc_lens[doc_idx]
                # BM25 term weighting
                tf_norm = (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                )
                scores[doc_idx] += term_idf * tf_norm

        # Find top_k scores > 0
        non_zero_indices = np.where(scores > 0)[0]
        if len(non_zero_indices) == 0:
            return []

        # Sort descending by score
        sorted_indices = non_zero_indices[np.argsort(scores[non_zero_indices])[::-1]]
        top_indices = sorted_indices[:top_k]

        results: List[Tuple[Chunk, float, int]] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append((self.chunks[idx], float(scores[idx]), rank))

        return results


class DenseIndex:
    """In-memory dense embedding index using fastembed BAAI/bge-small-en-v1.5.

    Embeddings are unit-normalized so inner product equals cosine similarity.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: Optional[np.ndarray] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        use_breadcrumb: bool = False,
        model: Optional[Any] = None,
    ) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self.model_name = model_name
        self.use_breadcrumb = use_breadcrumb
        self._model: Optional[Any] = model

        if embeddings is not None:
            if len(embeddings) != len(self.chunks):
                raise ValueError(
                    f"Embeddings count ({len(embeddings)}) does not match chunks count ({len(self.chunks)})"
                )
            self.embeddings = embeddings.astype(np.float32)
            self._normalize_embeddings()
        else:
            self.embeddings = np.empty((0, 384), dtype=np.float32)

    def _get_model(self) -> Any:
        """Lazy-load fastembed model."""
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _normalize_embeddings(self) -> None:
        """Ensure all stored embeddings are unit L2-normalized."""
        if self.embeddings.size == 0:
            return
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    def build_embeddings(self, batch_size: int = 64) -> np.ndarray:
        """Generate and normalize embeddings for all indexed chunks."""
        if not self.chunks:
            self.embeddings = np.empty((0, 384), dtype=np.float32)
            return self.embeddings

        model = self._get_model()
        texts = [c.to_retrieval_text(include_breadcrumb=self.use_breadcrumb) for c in self.chunks]
        raw_embs = list(model.embed(texts, batch_size=batch_size))
        self.embeddings = np.array(raw_embs, dtype=np.float32)
        self._normalize_embeddings()
        return self.embeddings

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float, int]]:
        """Search dense index by cosine similarity and return list of (Chunk, score, rank)."""
        if self.embeddings.size == 0 or len(self.chunks) == 0:
            return []

        model = self._get_model()
        # embed single query
        query_emb = list(model.embed([query]))[0]
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        # Matrix-vector dot product computes cosine similarities
        scores = np.dot(self.embeddings, query_emb)

        # Sort descending
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[Chunk, float, int]] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append((self.chunks[idx], float(scores[idx]), rank))

        return results


def save_index_cache(
    chunks: Sequence[Chunk],
    embeddings: Optional[np.ndarray],
    cache_dir: Path,
    prefix: str,
) -> None:
    """Save chunk metadata and embedding matrix to local disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = cache_dir / f"{prefix}_chunks.json"
    emb_path = cache_dir / f"{prefix}_embeddings.npy"

    chunks_data = [asdict(c) for c in chunks]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)

    if embeddings is not None and embeddings.size > 0:
        np.save(emb_path, embeddings)


def load_index_cache(
    cache_dir: Path,
    prefix: str,
) -> Tuple[List[Chunk], Optional[np.ndarray]]:
    """Load cached chunks and embeddings from local disk cache."""
    chunks_path = cache_dir / f"{prefix}_chunks.json"
    emb_path = cache_dir / f"{prefix}_embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks cache not found at {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [Chunk(**d) for d in chunks_data]

    embeddings: Optional[np.ndarray] = None
    if emb_path.exists():
        embeddings = np.load(emb_path)

    return chunks, embeddings

