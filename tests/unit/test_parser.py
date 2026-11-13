"""Unit tests for layout-aware PDF parser and table extraction."""

from pathlib import Path
import pytest

from relay.rag.parser import (
    LayoutAwarePDFParser,
    ParsedDocument,
    ParsedElement,
    format_table_as_grid,
    is_potential_heading,
)

PDF_ATTENTION = Path("data/corpus/pdfs/1706.03762v7.pdf")


def test_format_table_as_grid():
    """Verify table formatter builds structured markdown with preserved headers."""
    raw_table = [
        ["Model", "d_model", "Heads", "BLEU"],
        ["Base", "512", "8", "27.3"],
        ["Big", "1024", "16", "28.4"],
    ]
    grid = format_table_as_grid(raw_table, caption="Table 1: Translation results")
    assert "[CAPTION: Table 1: Translation results]" in grid
    assert "| Model | d_model | Heads | BLEU |" in grid
    assert "| --- | --- | --- | --- |" in grid
    assert "| Base | 512 | 8 | 27.3 |" in grid
    assert "| Big | 1024 | 16 | 28.4 |" in grid


def test_format_table_as_grid_empty():
    """Verify empty or None table rows return empty string gracefully."""
    assert format_table_as_grid([]) == ""
    assert format_table_as_grid([[None, None]]) == ""


def test_is_potential_heading():
    """Verify section heading detector identifies numbered and named headings."""
    assert is_potential_heading("1 Introduction") is True
    assert is_potential_heading("3.2 Multi-Head Attention") is True
    assert is_potential_heading("Abstract") is True
    assert is_potential_heading("5.1 Machine Translation") is True
    assert is_potential_heading("This is a normal paragraph sentence about transformers.") is False
    assert is_potential_heading("") is False


def test_parse_attention_paper():
    """Verify parser extracts layout-aware document elements from Attention PDF."""
    if not PDF_ATTENTION.exists():
        pytest.skip(f"Test PDF not present: {PDF_ATTENTION}")

    parser = LayoutAwarePDFParser()
    doc = parser.parse_pdf(PDF_ATTENTION, paper_id="1706.03762v7")

    assert doc.paper_id == "1706.03762v7"
    assert doc.version == "v7"
    assert doc.total_pages == 15
    assert len(doc.elements) > 50
    assert len(doc.sections) > 10

    # Verify presence of headings, paragraphs, and tables
    elem_types = {e.element_type for e in doc.elements}
    assert "paragraph" in elem_types
    assert "heading" in elem_types
    assert "table" in elem_types

    # Verify tables have structured table_data
    tables = [e for e in doc.elements if e.element_type == "table"]
    assert len(tables) > 0
    first_table = tables[0]
    assert first_table.table_data is not None
    assert len(first_table.table_data) > 0


def test_extract_roi():
    """Verify Region of Interest (ROI) cropping utility."""
    if not PDF_ATTENTION.exists():
        pytest.skip(f"Test PDF not present: {PDF_ATTENTION}")

    parser = LayoutAwarePDFParser()
    # Crop top of page 1 (title region: x0=50, top=50, x1=550, bottom=150)
    cropped_text = parser.extract_roi(PDF_ATTENTION, page_num=1, bbox=(50.0, 50.0, 550.0, 150.0))
    assert "Attention" in cropped_text or len(cropped_text) > 0

