"""Layout-aware PDF parsing for The Relay research paper corpus.

Extracts text blocks, headings, paragraphs, and tables with explicit bounding boxes,
reading-order detection, section hierarchy tracking, and exact page/section provenance.
Preserves table structures as structured grids rather than flattening them into raw text.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import os

import pdfplumber

_X_TOLERANCE = float(os.environ.get("RELAY_PDF_X_TOLERANCE", "1.5"))
_DROP_TABLE_TEXT = os.environ.get("RELAY_PDF_DROP_TABLE_TEXT", "1") == "1"

from relay.contracts.schemas import PaperIdType

# Regex patterns for detecting section headings
SECTION_HEADING_PATTERN = re.compile(
    r"^([0-9]+(\.[0-9]+)*\s+[A-Z][A-Za-z0-9\s,\-:]{2,70}|"
    r"Abstract|Introduction|Related Work|Background|Architecture|Model|"
    r"Experiments|Results|Discussion|Conclusion|Acknowledgments|References|"
    r"Appendix\s+[A-Z0-9]?(\s+.*)?)$",
    re.IGNORECASE,
)

TABLE_CAPTION_PATTERN = re.compile(
    r"^(Table\s+[0-9]+[A-Za-z]?\s*[:\.\-]\s*.*)",
    re.IGNORECASE,
)

FIGURE_CAPTION_PATTERN = re.compile(
    r"^(Figure\s+[0-9]+[A-Za-z]?\s*[:\.\-]\s*.*)",
    re.IGNORECASE,
)


@dataclass
class ParsedElement:
    """A structured element extracted from a layout-aware PDF parse."""

    element_id: str
    paper_id: PaperIdType
    version: str
    page: int
    section: str
    element_type: str  # "heading", "paragraph", "table", "caption"
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, top, x1, bottom)
    table_data: Optional[List[List[str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete layout-aware document parse with section tree and diagnostics."""

    paper_id: PaperIdType
    version: str
    title: str
    total_pages: int
    elements: List[ParsedElement]
    sections: List[str]
    warnings: List[str] = field(default_factory=list)


def format_table_as_grid(table_rows: List[List[Optional[str]]], caption: Optional[str] = None) -> str:
    """Format extracted table matrix into a structured markdown/grid representation.

    Guarantees column headers are retained and row groups remain contiguous.
    """
    if not table_rows:
        return ""

    # Clean None values and whitespace
    clean_rows: List[List[str]] = []
    for row in table_rows:
        cleaned_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        if any(cleaned_row):  # Skip completely empty rows
            clean_rows.append(cleaned_row)

    if not clean_rows:
        return ""

    lines: List[str] = []
    if caption:
        lines.append(f"[CAPTION: {caption}]")

    header = clean_rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in clean_rows[1:]:
        # Pad row to match header length if needed
        padded_row = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded_row[: len(header)]) + " |")

    return "\n".join(lines)


def is_potential_heading(line: str) -> bool:
    """Determine if a line of text represents a section heading."""
    clean = line.strip()
    if not clean or len(clean) > 80:
        return False
    if SECTION_HEADING_PATTERN.match(clean):
        return True
    # Catch numbered headings e.g. "3.2 Multi-Head Attention"
    parts = clean.split(maxsplit=1)
    if len(parts) == 2 and re.match(r"^[0-9]+(\.[0-9]+)*$", parts[0]):
        return True
    return False


class LayoutAwarePDFParser:
    """Parser that extracts research papers preserving layout, headings, and tables."""

    def __init__(
        self,
        paper_id: Optional[PaperIdType] = None,
        version: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        self.paper_id = paper_id
        self.version = version
        self.title = title
        self.warnings: List[str] = []

    def parse_pdf(
        self,
        pdf_path: Path,
        paper_id: Optional[PaperIdType] = None,
        version: Optional[str] = None,
        title: Optional[str] = None,
    ) -> ParsedDocument:
        """Parse an immutable PDF file into a structured ParsedDocument."""
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        # Resolve paper identity
        pid = paper_id or self.paper_id
        if not pid:
            stem = pdf_path.stem
            for known_id in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]:
                if stem.startswith(known_id) or known_id in stem:
                    pid = known_id  # type: ignore[assignment]
                    break

        if not pid:
            raise ValueError(f"Could not determine paper_id for {pdf_path}")

        from relay.rag.corpus import REQUIRED_PAPERS

        paper_meta = REQUIRED_PAPERS.get(pid)
        doc_version = version or self.version or (paper_meta.version if paper_meta else "v1")
        doc_title = title or self.title or (paper_meta.title if paper_meta else pid)

        elements: List[ParsedElement] = []
        observed_sections: List[str] = ["Title & Frontmatter"]
        current_section = "Title & Frontmatter"
        element_counter = 0

        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)

            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1  # 1-indexed

                # 1. Identify and extract tables
                tables_on_page = []
                table_bboxes = []
                try:
                    found_tables = page.find_tables()
                    for table_obj in found_tables:
                        extracted = table_obj.extract()
                        if extracted and len(extracted) > 1:
                            tables_on_page.append((table_obj.bbox, extracted))
                            table_bboxes.append(table_obj.bbox)
                except Exception as e:
                    self.warnings.append(f"Page {page_num}: Table detection error: {e}")

                # 2. Extract text outside tables (filter out table bounding boxes if any)
                try:
                    text_page = page
                    if table_bboxes and _DROP_TABLE_TEXT:
                        # Skip characters inside detected tables: they are already emitted as grid
                        # tables, and repeating them as prose created numbers-only junk chunks.
                        def _outside_tables(obj, _boxes=tuple(table_bboxes)):
                            if obj.get("object_type") != "char":
                                return True
                            return not any(obj["x0"] >= b[0] - 1 and obj["x1"] <= b[2] + 1 and
                                           obj["top"] >= b[1] - 1 and obj["bottom"] <= b[3] + 1 for b in _boxes)
                        text_page = page.filter(_outside_tables)

                    # Default x_tolerance=3 glued words in these LaTeX PDFs
                    # ("newstate-of-the-artBLEUscoreof28.4"); measured with scripts/check_extraction.py.
                    page_text = text_page.extract_text(layout=False, x_tolerance=_X_TOLERANCE) or ""
                    lines = page_text.split("\n")
                except Exception as e:
                    self.warnings.append(f"Page {page_num}: Text extraction error: {e}")
                    lines = []

                # 3. Process text lines into headings, captions, and paragraphs
                para_buffer: List[str] = []

                def flush_paragraph() -> None:
                    nonlocal element_counter
                    if para_buffer:
                        p_text = " ".join(para_buffer).strip()
                        if p_text:
                            element_counter += 1
                            elements.append(
                                ParsedElement(
                                    element_id=f"{self.paper_id}_p{page_num:02d}_e{element_counter:03d}",
                                    paper_id=self.paper_id,
                                    version=self.version,
                                    page=page_num,
                                    section=current_section,
                                    element_type="paragraph",
                                    text=p_text,
                                )
                            )
                        para_buffer.clear()

                for raw_line in lines:
                    line = raw_line.strip()
                    if not line:
                        flush_paragraph()
                        continue

                    # Filter out arXiv watermark header/footer
                    if "arXiv:" in line and ("[" in line or "http" in line):
                        continue
                    if re.match(r"^[0-9]+$", line):  # Standalone page numbers
                        continue

                    # Check for section heading
                    if is_potential_heading(line):
                        flush_paragraph()
                        current_section = line
                        if current_section not in observed_sections:
                            observed_sections.append(current_section)
                        element_counter += 1
                        elements.append(
                            ParsedElement(
                                element_id=f"{self.paper_id}_p{page_num:02d}_e{element_counter:03d}",
                                paper_id=self.paper_id,
                                version=self.version,
                                page=page_num,
                                section=current_section,
                                element_type="heading",
                                text=line,
                            )
                        )
                        continue

                    # Check for table or figure caption
                    if TABLE_CAPTION_PATTERN.match(line) or FIGURE_CAPTION_PATTERN.match(line):
                        flush_paragraph()
                        element_counter += 1
                        elements.append(
                            ParsedElement(
                                element_id=f"{self.paper_id}_p{page_num:02d}_e{element_counter:03d}",
                                paper_id=self.paper_id,
                                version=self.version,
                                page=page_num,
                                section=current_section,
                                element_type="caption",
                                text=line,
                            )
                        )
                        continue

                    para_buffer.append(line)

                flush_paragraph()

                # 4. Process tables as atomic structured elements
                for bbox, table_data in tables_on_page:
                    element_counter += 1
                    grid_text = format_table_as_grid(table_data)
                    if grid_text:
                        elements.append(
                            ParsedElement(
                                element_id=f"{self.paper_id}_p{page_num:02d}_t{element_counter:03d}",
                                paper_id=self.paper_id,
                                version=self.version,
                                page=page_num,
                                section=current_section,
                                element_type="table",
                                text=grid_text,
                                bbox=bbox,
                                table_data=[[str(c) if c is not None else "" for c in r] for r in table_data],
                                metadata={"rows": len(table_data), "cols": len(table_data[0]) if table_data else 0},
                            )
                        )

        return ParsedDocument(
            paper_id=pid,
            version=doc_version,
            title=doc_title,
            total_pages=total_pages,
            elements=elements,
            sections=observed_sections,
            warnings=self.warnings,
        )

    def extract_roi(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: Tuple[float, float, float, float],
    ) -> str:
        """Targeted Region of Interest (ROI) extraction fallback for dense boxes."""
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                raise ValueError(f"Page number {page_num} out of bounds")
            page = pdf.pages[page_num - 1]
            cropped = page.crop(bbox)
            return cropped.extract_text(x_tolerance=_X_TOLERANCE) or ""
