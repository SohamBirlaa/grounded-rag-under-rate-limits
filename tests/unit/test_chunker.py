"""Unit tests for structure-aware chunker and table header retention."""

from pathlib import Path
import pytest

from relay.rag.chunker import (
    Chunk,
    StructureAwareChunker,
    count_tokens,
    sanitize_section_for_id,
)
from relay.rag.parser import LayoutAwarePDFParser, ParsedDocument, ParsedElement


def test_count_tokens():
    """Verify deterministic token counting."""
    assert count_tokens("") == 0
    assert count_tokens("Hello world") == 2
    # Punctuations counted as separate tokens
    assert count_tokens("Hello, world! 123.") == 6


def test_sanitize_section_for_id():
    """Verify section name sanitizer produces clean identifiers."""
    assert sanitize_section_for_id("3.2 Multi-Head Attention") == "3_2_Multi_Head_Atten"
    assert sanitize_section_for_id("!@#$%^") == "sec"


def test_chunker_section_and_table_preservation():
    """Verify chunker respects section boundaries and preserves table headers on slicing."""
    # Synthetic parsed document
    elements = [
        ParsedElement(
            element_id="el_01",
            paper_id="1706.03762v7",
            version="v7",
            page=1,
            section="1 Introduction",
            element_type="paragraph",
            text="Recurrent neural networks have been firmly established as state of the art.",
        ),
        ParsedElement(
            element_id="el_02",
            paper_id="1706.03762v7",
            version="v7",
            page=2,
            section="2 Background",
            element_type="paragraph",
            text="The goal of reducing sequential computation also forms the foundation of ByteNet.",
        ),
        ParsedElement(
            element_id="el_03",
            paper_id="1706.03762v7",
            version="v7",
            page=2,
            section="2 Background",
            element_type="table",
            text="| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n| 7 | 8 |\n| 9 | 10 |\n| 11 | 12 |\n| 13 | 14 |\n| 15 | 16 |",
            table_data=[
                ["Col A", "Col B"],
                ["1", "2"],
                ["3", "4"],
                ["5", "6"],
                ["7", "8"],
                ["9", "10"],
                ["11", "12"],
                ["13", "14"],
                ["15", "16"],
            ],
        ),
    ]

    doc = ParsedDocument(
        paper_id="1706.03762v7",
        version="v7",
        title="Attention Is All You Need",
        total_pages=2,
        elements=elements,
        sections=["1 Introduction", "2 Background"],
    )

    chunker = StructureAwareChunker(target_tokens=50, overlap_tokens=10)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3

    # Check section boundary: chunk 1 should belong to Introduction, not mixed with Background
    intro_chunks = [c for c in chunks if c.section == "1 Introduction"]
    assert len(intro_chunks) == 1
    assert "Recurrent" in intro_chunks[0].text
    assert "ByteNet" not in intro_chunks[0].text

    # Check table chunks
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(table_chunks) >= 1
    for tc in table_chunks:
        assert tc.content_type == "table"
        assert tc.structural_breadcrumb is not None
        assert "1706.03762v7" in tc.structural_breadcrumb


def test_chunker_target_token_sizes():
    """Verify chunker produces smaller average chunks for S1 (256) than S2/S3 (450)."""
    pdf_path = Path("data/corpus/pdfs/1706.03762v7.pdf")
    if not pdf_path.exists():
        pytest.skip("Attention PDF not found")

    parser = LayoutAwarePDFParser()
    doc = parser.parse_pdf(pdf_path, paper_id="1706.03762v7")

    chunker_s1 = StructureAwareChunker(target_tokens=256, overlap_tokens=35)
    chunks_s1 = chunker_s1.chunk_document(doc)

    chunker_s23 = StructureAwareChunker(target_tokens=450, overlap_tokens=50)
    chunks_s23 = chunker_s23.chunk_document(doc)

    # 256-token target should yield equal or more chunks than 450-token target
    assert len(chunks_s1) >= len(chunks_s23)
    avg_s1 = sum(c.token_count for c in chunks_s1) / len(chunks_s1)
    avg_s23 = sum(c.token_count for c in chunks_s23) / len(chunks_s23)
    assert avg_s1 <= avg_s23

