"""Unit tests for ContextBuilder and prompt assembly."""

import pytest

from relay.rag.context_builder import ContextBuilder, GROUNDING_SYSTEM_PROMPT
from relay.rag.retriever import RetrievedChunk


@pytest.fixture
def sample_retrieved_chunks():
    return [
        RetrievedChunk(
            chunk_id="1706.03762v7_sec_3_2_p04_c014",
            paper_id="1706.03762v7",
            version="v7",
            page=4,
            section="3.2.1 Scaled Dot-Product Attention",
            content_type="prose",
            text="We compute the attention function on a set of queries simultaneously.",
            token_count=12,
            score=0.032,
            rank=1,
            retrieval_method="hybrid_rrf",
            structural_breadcrumb="[Doc: Attention | Page 4]",
        ),
        RetrievedChunk(
            chunk_id="1706.03762v7_sec_3_2_2_p05_c015",
            paper_id="1706.03762v7",
            version="v7",
            page=5,
            section="3.2.2 Multi-Head Attention",
            content_type="prose",
            text="Multi-head attention allows the model to jointly attend to information.",
            token_count=11,
            score=0.031,
            rank=2,
            retrieval_method="hybrid_rrf",
            structural_breadcrumb="[Doc: Attention | Page 5]",
        ),
    ]


def test_build_passages_xml(sample_retrieved_chunks):
    """Verify XML passage formatting contains all required provenance attributes."""
    cb = ContextBuilder()
    xml = cb.build_passages_xml(sample_retrieved_chunks)

    assert "<retrieved_passages>" in xml
    assert "</retrieved_passages>" in xml
    assert '<passage id="1706.03762v7_sec_3_2_p04_c014"' in xml
    assert 'paper_id="1706.03762v7"' in xml
    assert 'version="v7"' in xml
    assert 'page="4"' in xml
    assert 'section="3.2.1 Scaled Dot-Product Attention"' in xml
    assert "We compute the attention function" in xml


def test_build_passages_xml_empty():
    """Verify empty passages gracefully outputs empty XML container."""
    cb = ContextBuilder()
    xml = cb.build_passages_xml([])
    assert "<retrieved_passages>" in xml
    assert "<!-- No relevant passages retrieved -->" in xml


def test_build_messages_structure(sample_retrieved_chunks):
    """Verify build_messages constructs standard system and user chat payload."""
    cb = ContextBuilder()
    messages = cb.build_messages("What is Scaled Dot-Product Attention?", sample_retrieved_chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "EPISTEMIC ABSTENTION RULE" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "What is Scaled Dot-Product Attention?" in messages[1]["content"]
    assert "<retrieved_passages>" in messages[1]["content"]


def test_estimate_prompt_tokens(sample_retrieved_chunks):
    """Verify token estimation accounts for system prompt, passages, and query."""
    cb = ContextBuilder()
    token_est = cb.estimate_prompt_tokens("What is attention?", sample_retrieved_chunks)
    assert token_est > 100

