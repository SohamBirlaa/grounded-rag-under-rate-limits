"""RAG subsystem for The Relay.

Provides:
- Paper metadata and verification: CorpusManager, PaperMetadata
- Layout-aware PDF parsing: LayoutAwarePDFParser, ParsedDocument, ParsedElement
- Structure-aware chunking: StructureAwareChunker, Chunk, count_tokens
- In-memory indexing: BM25Index, DenseIndex, save_index_cache, load_index_cache
- Multi-setting retrieval: Retriever, RetrievedChunk
- Context building and prompt assembly: ContextBuilder, GROUNDING_SYSTEM_PROMPT
"""

from relay.rag.corpus import CorpusManager, PaperMetadata, CORPUS_PAPERS
from relay.rag.parser import LayoutAwarePDFParser, ParsedDocument, ParsedElement, format_table_as_grid
from relay.rag.chunker import StructureAwareChunker, Chunk, count_tokens
from relay.rag.index import BM25Index, DenseIndex, save_index_cache, load_index_cache
from relay.rag.retriever import Retriever, RetrievedChunk
from relay.rag.context_builder import ContextBuilder, GROUNDING_SYSTEM_PROMPT

__all__ = [
    "CorpusManager",
    "PaperMetadata",
    "CORPUS_PAPERS",
    "LayoutAwarePDFParser",
    "ParsedDocument",
    "ParsedElement",
    "format_table_as_grid",
    "StructureAwareChunker",
    "Chunk",
    "count_tokens",
    "BM25Index",
    "DenseIndex",
    "save_index_cache",
    "load_index_cache",
    "Retriever",
    "RetrievedChunk",
    "ContextBuilder",
    "GROUNDING_SYSTEM_PROMPT",
]

