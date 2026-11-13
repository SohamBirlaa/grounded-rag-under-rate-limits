"""Unit and latency tests for the multi-setting Retriever."""

from pathlib import Path
import time
import pytest

from relay.rag.retriever import Retriever, RetrievedChunk

INDEXES_DIR = Path("data/corpus/indexes")


@pytest.fixture(scope="module")
def retriever():
    """Load retriever from generated index cache."""
    if not (INDEXES_DIR / "s1_256_chunks.json").exists():
        pytest.skip(f"Index cache not found in {INDEXES_DIR}")
    return Retriever.from_cache(INDEXES_DIR)


def test_retriever_s1_bm25_top3(retriever):
    """Verify S1 retrieval returns exactly 3 pieces via BM25."""
    results = retriever.retrieve_passages(
        query="Transformer self-attention scaled dot-product",
        setting="S1_256_3_no",
    )
    assert len(results) == 3
    for rank, r in enumerate(results, start=1):
        assert isinstance(r, RetrievedChunk)
        assert r.retrieval_method == "bm25"
        assert r.rank == rank
        assert r.paper_id in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]
        assert r.page > 0
        assert len(r.text) > 0


def test_retriever_s2_dense_top5(retriever):
    """Verify S2 retrieval returns exactly 5 pieces via BGE-small dense search."""
    results = retriever.retrieve_passages(
        query="Mimi neural audio codec frame rate 12.5 Hz",
        setting="S2_450_5_no",
    )
    assert len(results) == 5
    for rank, r in enumerate(results, start=1):
        assert isinstance(r, RetrievedChunk)
        assert r.retrieval_method == "dense"
        assert r.rank == rank


def test_retriever_s3_hybrid_rrf_top4(retriever):
    """Verify S3 retrieval executes RRF fusion and selects top 4 deduplicated pieces."""
    results = retriever.retrieve_passages(
        query="dynamic chunking hierarchical sequence modeling mamba",
        setting="S3_450_8to4_rrf_no",
    )
    assert len(results) == 4

    # Verify no duplicate chunk IDs
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))

    for rank, r in enumerate(results, start=1):
        assert r.retrieval_method == "hybrid_rrf"
        assert r.rank == rank
        assert r.score > 0.0


def test_retriever_compact_mode_override(retriever):
    """Verify Quota-Conserving Compact Mode forces k=2 and token truncation."""
    results = retriever.retrieve_passages(
        query="multi-head attention layers",
        setting="S3_450_8to4_rrf_no",
        compact_mode=True,
    )
    assert len(results) == 2
    for r in results:
        # Each chunk should have token count within compact limit or truncated
        assert r.token_count <= 210  # 200 tokens + truncation notice margin


def test_retriever_latency_under_15ms(retriever):
    """Verify that in-memory retrieval completes in under 15ms per call on local CPU."""
    # Warm-up call
    retriever.retrieve_passages("test query", setting="S1_256_3_no")

    t0 = time.perf_counter()
    n_queries = 20
    for i in range(n_queries):
        retriever.retrieve_passages("attention mechanism query", setting="S1_256_3_no")
    elapsed_total = time.perf_counter() - t0
    avg_latency_ms = (elapsed_total / n_queries) * 1000.0

    print(f"\nAverage S1 retrieval latency: {avg_latency_ms:.2f} ms")
    assert avg_latency_ms < 15.0, f"Retrieval latency too slow: {avg_latency_ms:.2f} ms"

