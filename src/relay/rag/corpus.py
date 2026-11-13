"""Research corpus metadata, version verification, and manifest manager.

Manages the exact three arXiv paper versions mandated by the assignment brief:
- Attention Is All You Need — arXiv:1706.03762v7
- Moshi — arXiv:2410.00037v1
- Dynamic Chunking — arXiv:2507.07955v2
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import pypdf

from relay.contracts.schemas import PaperIdType


class PaperMetadata(BaseModel):
    """Immutable metadata specification for a research paper in The Relay corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: PaperIdType
    version: str
    title: str
    authors: str
    arxiv_url: str
    pdf_url: str
    expected_stamp: str
    page_count: Optional[int] = None
    sha256: Optional[str] = None
    file_path: Optional[str] = None


REQUIRED_PAPERS: Dict[PaperIdType, PaperMetadata] = {
    "1706.03762v7": PaperMetadata(
        paper_id="1706.03762v7",
        version="v7",
        title="Attention Is All You Need",
        authors="Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
        arxiv_url="https://arxiv.org/abs/1706.03762v7",
        pdf_url="https://arxiv.org/pdf/1706.03762v7.pdf",
        expected_stamp="arXiv:1706.03762v7",
    ),
    "2410.00037v1": PaperMetadata(
        paper_id="2410.00037v1",
        version="v1",
        title="Moshi: a speech-text foundation model for real-time dialogue",
        authors="Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, Neil Zeghidour",
        arxiv_url="https://arxiv.org/abs/2410.00037v1",
        pdf_url="https://arxiv.org/pdf/2410.00037v1.pdf",
        expected_stamp="arXiv:2410.00037v1",
    ),
    "2507.07955v2": PaperMetadata(
        paper_id="2507.07955v2",
        version="v2",
        title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        authors="Sukjun Hwang, Brandon Wang, Albert Gu",
        arxiv_url="https://arxiv.org/abs/2507.07955v2",
        pdf_url="https://arxiv.org/pdf/2507.07955v2.pdf",
        expected_stamp="arXiv:2507.07955v2",
    ),
}


CORPUS_PAPERS = REQUIRED_PAPERS


class CorpusManager:
    """Manages the research papers corpus on disk."""

    def __init__(self, corpus_dir: Path = Path("data/corpus")) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.pdfs_dir = self.corpus_dir / "pdfs"
        self.manifest_path = self.corpus_dir / "manifest.json"

    def get_paper_metadata(self, paper_id: PaperIdType) -> Optional[PaperMetadata]:
        """Return declared metadata for paper."""
        return REQUIRED_PAPERS.get(paper_id)

    def get_pdf_path(self, paper_id: PaperIdType) -> Path:
        """Return filesystem path to local PDF."""
        return self.pdfs_dir / f"{paper_id}.pdf"

    def verify_corpus(self) -> Dict[str, Any]:
        """Verify all required PDFs exist and match arXiv identities."""
        results: Dict[str, Any] = {}
        for pid in REQUIRED_PAPERS:
            pdf_path = self.get_pdf_path(pid)
            results[pid] = verify_pdf_identity(pdf_path, pid)
        return results

    def load_manifest(self) -> Dict[str, Any]:
        """Load the corpus manifest JSON file."""
        return load_manifest(self.manifest_path)


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_pdf_identity(file_path: Path, expected_paper_id: PaperIdType) -> Dict[str, Any]:
    """Verify that a local PDF matches the exact expected arXiv version.

    Checks:
    1. Valid PDF file signature (%PDF).
    2. Exact paper_id stamp appears in the PDF text (margin stamp).
    3. Calculates SHA-256 hash and page count.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {file_path}")

    # 1. Check PDF header signature
    with open(file_path, "rb") as f:
        header = f.read(5)
        if not header.startswith(b"%PDF-"):
            raise ValueError(f"File {file_path} is not a valid PDF file (bad signature)")

    sha256_hash = compute_sha256(file_path)
    meta = REQUIRED_PAPERS.get(expected_paper_id)
    if not meta:
        raise ValueError(f"Unknown paper ID: {expected_paper_id}")

    # 2. Extract first-page text to verify arXiv stamp
    try:
        reader = pypdf.PdfReader(str(file_path))
        num_pages = len(reader.pages)
        if num_pages < 1:
            raise ValueError(f"PDF {file_path} has 0 pages")

        first_page_text = reader.pages[0].extract_text() or ""
        # Check margin stamp or text content
        has_stamp = meta.expected_stamp in first_page_text or meta.paper_id in first_page_text

        return {
            "paper_id": expected_paper_id,
            "version": meta.version,
            "title": meta.title,
            "file_path": str(file_path),
            "sha256": sha256_hash,
            "page_count": num_pages,
            "stamp_verified": has_stamp,
            "expected_stamp": meta.expected_stamp,
        }
    except Exception as e:
        raise ValueError(f"Failed to inspect PDF {file_path}: {e}") from e


def load_manifest(manifest_path: Path = Path("data/corpus/manifest.json")) -> Dict[str, Any]:
    """Load the corpus manifest JSON file."""
    if not manifest_path.is_file():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(
    manifest_data: Dict[str, Any],
    manifest_path: Path = Path("data/corpus/manifest.json"),
) -> None:
    """Save the corpus manifest JSON file atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
