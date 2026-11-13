"""Unit tests for BM25 and Dense vector indexing."""

from pathlib import Path
import numpy as np
import pytest

from relay.rag.chunker import Chunk
from relay.rag.index import (
    BM25Index,
    DenseIndex,
    load_index_cache,
    save_index_cache,
    tokenize_text,
)


@pytest.fixture
def sample_chunks():
    """Create a minimal set of test chunks."""
    return [
        Chunk(
            chunk_id="chunk_01",
            paper_id="1706.03762v7",
            version="v7",
            page=1,
            section="1 Introduction",
            content_type="prose",
            text="The Transformer is the first transduction model relying entirely on self-attention.",
            token_count=13,
            structural_breadcrumb="[Doc: Attention | Page 1]",
        ),
        Chunk(
            chunk_id="chunk_02",
            paper_id="2410.00037v1",
            version="v1",
            page=4,
            section="3.1 Mimi",
            content_type="prose",
            text="Mimi is a neural audio codec operating at a frame rate of 12.5 Hz.",
            token_count=15,
            structural_breadcrumb="[Doc: Moshi | Page 4]",
        ),
        Chunk(
            chunk_id="chunk_03",
            paper_id="2507.07955v2",
            version="v2",
            page=2,
            section="1 Introduction",
            content_type="prose",
            text="Dynamic chunking enables end-to-end hierarchical sequence modeling on byte sequences.",
            token_count=12,
            structural_breadcrumb="[Doc: Dynamic Chunking | Page 2]",
        ),
    ]


def test_tokenize_text():
    """Verify alphanumeric lowercase tokenization."""
    tokens = tokenize_text("Transformer: Self-Attention & 12.5 Hz!")
    assert "transformer" in tokens
    assert "self" in tokens
    assert "attention" in tokens
    assert "12" in tokens
    assert "5" in tokens
    assert "hz" in tokens


def test_bm25_search(sample_chunks):
    """Verify BM25 ranking finds exact keyword matches."""
    index = BM25Index(sample_chunks)
    assert index.n_docs == 3

    # Query for self-attention
    results = index.search("self-attention transduction model", top_k=2)
    assert len(results) >= 1
    top_chunk, score, rank = results[0]
    assert top_chunk.chunk_id == "chunk_01"
    assert rank == 1
    assert score > 0.0

    # Query for audio codec
    results_mimi = index.search("mimi audio codec 12.5 Hz", top_k=2)
    assert len(results_mimi) >= 1
    assert results_mimi[0][0].chunk_id == "chunk_02"


def test_bm25_empty_query_or_no_match(sample_chunks):
    """Verify BM25 returns empty list on irrelevant or empty queries."""
    index = BM25Index(sample_chunks)
    assert index.search("") == []
    assert index.search("xylophone quantum entanglement") == []


def test_dense_index_with_precomputed_embeddings(sample_chunks):
    """Verify dense cosine search with synthetic precomputed embeddings."""
    # 3 chunks, 4 dimensions
    raw_embs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # chunk_01
            [0.0, 1.0, 0.0, 0.0],  # chunk_02
            [0.0, 0.0, 1.0, 0.0],  # chunk_03
        ],
        dtype=np.float32,
    )
    dense_idx = DenseIndex(sample_chunks, embeddings=raw_embs)
    assert dense_idx.embeddings.shape == (3, 4)

    # Unit normalization verified
    for row in dense_idx.embeddings:
        assert np.isclose(np.linalg.norm(row), 1.0)


def test_save_and_load_index_cache(tmp_path, sample_chunks):
    """Verify serialization and deserialization of index cache."""
    embs = np.random.randn(len(sample_chunks), 384).astype(np.float32)
    save_index_cache(sample_chunks, embs, cache_dir=tmp_path, prefix="test_cache")

    loaded_chunks, loaded_embs = load_index_cache(cache_dir=tmp_path, prefix="test_cache")
    assert len(loaded_chunks) == len(sample_chunks)
    assert loaded_chunks[0].chunk_id == sample_chunks[0].chunk_id
    assert loaded_chunks[1].text == sample_chunks[1].text
    assert loaded_embs is not None
    assert loaded_embs.shape == (3, 384)
    assert np.allclose(embs, loaded_embs)

