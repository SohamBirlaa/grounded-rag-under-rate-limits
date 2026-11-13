"""Unit tests for research corpus verification and metadata integrity."""

from pathlib import Path
import pytest

from relay.rag.corpus import (
    CorpusManager,
    PaperMetadata,
    REQUIRED_PAPERS,
    compute_sha256,
    verify_pdf_identity,
)

CORPUS_DIR = Path("data/corpus")
PDFS_DIR = CORPUS_DIR / "pdfs"


def test_required_papers_definition():
    """Verify all 3 papers are declared with exact arXiv IDs and versions."""
    assert "1706.03762v7" in REQUIRED_PAPERS
    assert "2410.00037v1" in REQUIRED_PAPERS
    assert "2507.07955v2" in REQUIRED_PAPERS

    assert REQUIRED_PAPERS["1706.03762v7"].version == "v7"
    assert REQUIRED_PAPERS["2410.00037v1"].version == "v1"
    assert REQUIRED_PAPERS["2507.07955v2"].version == "v2"


def test_local_pdf_files_exist_and_match_sha256():
    """Verify all 3 downloaded PDFs exist and match verified SHA-256 hashes."""
    expected_hashes = {
        "1706.03762v7": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
        "2410.00037v1": "a687144cc77bea86b0c60ae800ccb1d3e0816033fa5c7960388fe29a0dbfcfac",
        "2507.07955v2": "04a2b32e6e82853cfc23c5db132750874a86cde99219ec6f664128f4aa83122f",
    }

    for paper_id, expected_hash in expected_hashes.items():
        pdf_path = PDFS_DIR / f"{paper_id}.pdf"
        assert pdf_path.is_file(), f"Missing PDF for {paper_id}"
        computed_hash = compute_sha256(pdf_path)
        assert computed_hash == expected_hash, f"SHA-256 mismatch for {paper_id}"


def test_verify_pdf_identity_margin_stamps():
    """Verify that arXiv margin stamps are confirmed on the first page of each PDF."""
    for paper_id in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]:
        pdf_path = PDFS_DIR / f"{paper_id}.pdf"
        info = verify_pdf_identity(pdf_path, paper_id)
        assert info["stamp_verified"] is True
        assert info["page_count"] > 0
        assert info["paper_id"] == paper_id


def test_corpus_manager_helpers():
    """Verify CorpusManager helper methods."""
    manager = CorpusManager(corpus_dir=CORPUS_DIR)
    manifest = manager.load_manifest()
    assert "1706.03762v7" in manifest
    assert "2410.00037v1" in manifest
    assert "2507.07955v2" in manifest

    pdf_path = manager.get_pdf_path("1706.03762v7")
    assert pdf_path.exists()

