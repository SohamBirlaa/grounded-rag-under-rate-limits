
"""Offline corpus indexing script for The Relay.

Parses all 3 verified research PDFs:
- Attention Is All You Need (1706.03762v7)
- Moshi (2410.00037v1)
- Dynamic Chunking (2507.07955v2)

Generates layout-aware, structure-preserving chunks:
- S1: 256-token chunks (~35 overlap) for BM25
- S2/S3: 450-token chunks (~50 overlap) for Dense & Hybrid RRF

Precomputes unit-normalized BAAI/bge-small-en-v1.5 embeddings.
Saves precomputed artifacts to data/corpus/indexes/ for sub-10ms test runs and offline serving.
"""

import json
import logging
from pathlib import Path
import sys
import time

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from relay.rag.corpus import CorpusManager
from relay.rag.parser import LayoutAwarePDFParser
from relay.rag.chunker import StructureAwareChunker
from relay.rag.index import DenseIndex, save_index_cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_indexes")


def main() -> None:
    corpus_dir = REPO_ROOT / "data" / "corpus"
    indexes_dir = corpus_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    manager = CorpusManager(corpus_dir=corpus_dir)
    parser = LayoutAwarePDFParser()

    # 1. Parse all 3 papers
    logger.info("Parsing 3 research papers...")
    parsed_docs = []
    for paper_id in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]:
        pdf_path = corpus_dir / "pdfs" / f"{paper_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF missing: {pdf_path}")
        logger.info(f"Parsing {paper_id} from {pdf_path}...")
        t0 = time.perf_counter()
        doc = parser.parse_pdf(pdf_path, paper_id=paper_id)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Parsed {paper_id}: {doc.total_pages} pages, {len(doc.elements)} elements, "
            f"{len(doc.sections)} sections in {elapsed:.2f}s"
        )
        parsed_docs.append(doc)

    # 2. Chunk for S1 (target 256 tokens)
    logger.info("Chunking for S1 (target 256 tokens)...")
    chunker_256 = StructureAwareChunker(target_tokens=256, overlap_tokens=35)
    chunks_256 = []
    for doc in parsed_docs:
        doc_chunks = chunker_256.chunk_document(doc)
        logger.info(f"Paper {doc.paper_id}: {len(doc_chunks)} chunks (256-token target)")
        chunks_256.extend(doc_chunks)
    logger.info(f"Total S1 chunks: {len(chunks_256)}")

    # 3. Chunk for S2/S3 (target 450 tokens)
    logger.info("Chunking for S2/S3 (target 450 tokens)...")
    chunker_450 = StructureAwareChunker(target_tokens=450, overlap_tokens=50)
    chunks_450 = []
    for doc in parsed_docs:
        doc_chunks = chunker_450.chunk_document(doc)
        logger.info(f"Paper {doc.paper_id}: {len(doc_chunks)} chunks (450-token target)")
        chunks_450.extend(doc_chunks)
    logger.info(f"Total S2/S3 chunks: {len(chunks_450)}")

    # 4. Generate BGE-small embeddings for 450-token chunks
    logger.info("Computing BAAI/bge-small-en-v1.5 embeddings for 450-token chunks...")
    t0 = time.perf_counter()
    dense_idx = DenseIndex(chunks_450, model_name="BAAI/bge-small-en-v1.5")
    embs_450 = dense_idx.build_embeddings(batch_size=32)
    elapsed = time.perf_counter() - t0
    logger.info(f"Computed embeddings shape {embs_450.shape} in {elapsed:.2f}s")

    # 5. Save index cache files
    logger.info(f"Saving index cache to {indexes_dir}...")
    save_index_cache(chunks_256, embeddings=None, cache_dir=indexes_dir, prefix="s1_256")
    save_index_cache(chunks_450, embeddings=embs_450, cache_dir=indexes_dir, prefix="s23_450")

    # 6. Generate index manifest
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "papers": [
            {
                "paper_id": d.paper_id,
                "version": d.version,
                "title": d.title,
                "pages": d.total_pages,
                "elements_count": len(d.elements),
                "sections_count": len(d.sections),
            }
            for d in parsed_docs
        ],
        "s1_256": {
            "total_chunks": len(chunks_256),
            "avg_tokens": sum(c.token_count for c in chunks_256) / max(1, len(chunks_256)),
            "tables_count": sum(1 for c in chunks_256 if c.content_type == "table"),
        },
        "s23_450": {
            "total_chunks": len(chunks_450),
            "avg_tokens": sum(c.token_count for c in chunks_450) / max(1, len(chunks_450)),
            "tables_count": sum(1 for c in chunks_450 if c.content_type == "table"),
            "embedding_dims": int(embs_450.shape[1]),
        },
    }

    manifest_path = indexes_dir / "index_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Index build complete. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

