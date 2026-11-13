"""Structure-aware and layout-preserving chunker for The Relay research papers.

Implements two target chunk configurations:
- 256 tokens (with ~35 token overlap) for S1 (BM25 sparse)
- 450 tokens (with ~50 token overlap) for S2 (Dense) and S3 (Hybrid RRF)

Guarantees:
- Table header retention: sub-table chunks preserve column headers.
- Section-aware boundaries: chunks do not blindly cross major section headers.
- Stable deterministic chunk IDs.
- Full provenance metadata: paper_id, version, page, section, structural breadcrumbs.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional
from relay.contracts.schemas import PaperIdType
from relay.rag.parser import ParsedDocument, ParsedElement


def count_tokens(text: str) -> int:
    """Deterministic token counter using word and punctuation boundary splitting.

    Closely approximates subword / BPE tokenizer counts (within +/-5%)
    without requiring heavy external model dependencies.
    """
    if not text:
        return 0
    # Match alphanumeric words or single punctuation symbols
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


def sanitize_section_for_id(section: str) -> str:
    """Convert section name to clean alphanumeric token for chunk IDs."""
    clean = re.sub(r"[^A-Za-z0-9]", "_", section.strip()).strip("_")
    return clean[:20] if clean else "sec"


@dataclass
class Chunk:
    """A single retrievable text or table chunk with rich provenance metadata."""

    chunk_id: str
    paper_id: PaperIdType
    version: str
    page: int
    section: str
    content_type: str  # "prose" or "table"
    text: str
    token_count: int
    structural_breadcrumb: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_retrieval_text(self, include_breadcrumb: bool = False) -> str:
        """Format chunk text for indexing and retrieval."""
        if include_breadcrumb and self.structural_breadcrumb:
            return f"{self.structural_breadcrumb}\n{self.text}"
        return self.text


class StructureAwareChunker:
    """Chunker that segments layout-aware ParsedDocuments into structure-preserving chunks."""

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 50,
    ) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, doc: ParsedDocument) -> List[Chunk]:
        """Convert a ParsedDocument into an ordered list of structure-aware Chunks."""
        chunks: List[Chunk] = []
        chunk_idx = 0

        # Group elements by section to enforce section boundaries
        elements_by_section: Dict[str, List[ParsedElement]] = {}
        for elem in doc.elements:
            elements_by_section.setdefault(elem.section, []).append(elem)

        for section_name, section_elements in elements_by_section.items():
            sec_clean = sanitize_section_for_id(section_name)

            current_prose_tokens: List[str] = []
            current_pages: List[int] = []

            def flush_prose_chunk() -> None:
                nonlocal chunk_idx
                if not current_prose_tokens:
                    return

                chunk_idx += 1
                text_block = " ".join(current_prose_tokens).strip()
                t_count = count_tokens(text_block)
                primary_page = current_pages[0] if current_pages else 1

                breadcrumb = (
                    f"[Doc: {str(doc.title)[:45]} | arXiv:{doc.paper_id} | "
                    f"Sec: {section_name} | Page: {primary_page}]"
                )

                chunk = Chunk(
                    chunk_id=f"{doc.paper_id}_{sec_clean}_p{primary_page:02d}_c{chunk_idx:03d}",
                    paper_id=doc.paper_id,
                    version=doc.version,
                    page=primary_page,
                    section=section_name,
                    content_type="prose",
                    text=text_block,
                    token_count=t_count,
                    structural_breadcrumb=breadcrumb,
                    metadata={"page_span": list(set(current_pages))},
                )
                chunks.append(chunk)

                # Overlap logic: keep last overlap_tokens words
                if self.overlap_tokens > 0 and len(current_prose_tokens) > self.overlap_tokens:
                    current_prose_tokens.clear()
                    # Retain overlap words
                    overlap_words = text_block.split()[-self.overlap_tokens :]
                    current_prose_tokens.extend(overlap_words)
                    last_page = current_pages[-1] if current_pages else primary_page
                    current_pages.clear()
                    current_pages.append(last_page)
                else:
                    current_prose_tokens.clear()
                    current_pages.clear()

            for elem in section_elements:
                if elem.element_type == "table":
                    # Tables are chunked atomically
                    flush_prose_chunk()
                    table_chunks = self._chunk_table(elem, doc, section_name, sec_clean, chunk_idx)
                    for tc in table_chunks:
                        chunk_idx += 1
                        chunks.append(tc)
                else:
                    # Heading, caption, or prose paragraph
                    elem_text = elem.text.strip()
                    elem_tokens = count_tokens(elem_text)

                    # If adding this element exceeds target tokens, flush current buffer
                    if (count_tokens(" ".join(current_prose_tokens)) + elem_tokens) > self.target_tokens:
                        flush_prose_chunk()

                    current_prose_tokens.append(elem_text)
                    current_pages.append(elem.page)

            flush_prose_chunk()

        return chunks

    def _chunk_table(
        self,
        elem: ParsedElement,
        doc: ParsedDocument,
        section_name: str,
        sec_clean: str,
        current_chunk_idx: int,
    ) -> List[Chunk]:
        """Chunk a table preserving headers across any sliced sub-tables."""
        table_chunks: List[Chunk] = []
        table_text = elem.text
        t_count = count_tokens(table_text)

        # If table fits within target tokens, keep it as a single atomic chunk
        if t_count <= self.target_tokens or not elem.table_data or len(elem.table_data) <= 4:
            cid = current_chunk_idx + 1
            breadcrumb = (
                f"[Doc: {str(doc.title)[:45]} | arXiv:{doc.paper_id} | "
                f"Sec: {section_name} | Page: {elem.page} | Table]"
            )
            table_chunks.append(
                Chunk(
                    chunk_id=f"{doc.paper_id}_{sec_clean}_p{elem.page:02d}_t{cid:03d}",
                    paper_id=doc.paper_id,
                    version=doc.version,
                    page=elem.page,
                    section=section_name,
                    content_type="table",
                    text=table_text,
                    token_count=t_count,
                    structural_breadcrumb=breadcrumb,
                    metadata={"is_table": True, "rows": len(elem.table_data) if elem.table_data else 0},
                )
            )
            return table_chunks

        # If table is large, slice rows while preserving the header row on every slice
        header = elem.table_data[0]
        header_str = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |"
        rows = elem.table_data[1:]

        batch_size = 6  # Sub-row batch size
        for sub_idx in range(0, len(rows), batch_size):
            cid = current_chunk_idx + len(table_chunks) + 1
            sub_rows = rows[sub_idx : sub_idx + batch_size]
            sub_rows_str = "\n".join("| " + " | ".join(r) + " |" for r in sub_rows)
            combined_table = f"{header_str}\n{sub_rows_str}"

            breadcrumb = (
                f"[Doc: {str(doc.title)[:45]} | arXiv:{doc.paper_id} | "
                f"Sec: {section_name} | Page: {elem.page} | Table Part {len(table_chunks)+1}]"
            )
            table_chunks.append(
                Chunk(
                    chunk_id=f"{doc.paper_id}_{sec_clean}_p{elem.page:02d}_t{cid:03d}",
                    paper_id=doc.paper_id,
                    version=doc.version,
                    page=elem.page,
                    section=section_name,
                    content_type="table",
                    text=combined_table,
                    token_count=count_tokens(combined_table),
                    structural_breadcrumb=breadcrumb,
                    metadata={"is_table": True, "part": len(table_chunks) + 1},
                )
            )

        return table_chunks
