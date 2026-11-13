"""Download and verify the exact three arXiv research papers for The Relay.

Paper versions locked:
1. Attention Is All You Need — arXiv:1706.03762v7
2. Moshi: a speech-text foundation model — arXiv:2410.00037v1
3. Dynamic Chunking — arXiv:2507.07955v2
"""

import os
from pathlib import Path
import sys
import time
import urllib.request

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from relay.rag.corpus import REQUIRED_PAPERS, save_manifest, verify_pdf_identity


def fetch_paper(paper_id: str, dest_path: Path) -> None:
    """Download a single paper from arXiv with friendly headers and retry."""
    meta = REQUIRED_PAPERS[paper_id]  # type: ignore[index]
    url = meta.pdf_url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Research Corpus Ingestion; The Relay)",
    }

    print(f"Fetching {paper_id} ({meta.title[:40]}...) from {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers=headers)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                chunk_size = 65536
                while chunk := response.read(chunk_size):
                    out_file.write(chunk)
            print(f" Successfully downloaded {dest_path.name} ({dest_path.stat().st_size / 1024:.1f} KB)")
            return
        except Exception as e:
            print(f"  Attempt {attempt}/{max_retries} failed for {paper_id}: {e}")
            if attempt < max_retries:
                time.sleep(2.0 * attempt)
            else:
                raise RuntimeError(f"Failed to download {paper_id} after {max_retries} attempts: {e}") from e


def main() -> None:
    corpus_dir = Path("data/corpus/pdfs")
    manifest_path = Path("data/corpus/manifest.json")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = {}

    for paper_id, meta in REQUIRED_PAPERS.items():
        pdf_file = corpus_dir / f"{paper_id}.pdf"
        if not pdf_file.is_file() or pdf_file.stat().st_size < 10000:
            fetch_paper(paper_id, pdf_file)
            time.sleep(1.0)  # Gentle pacing on arXiv
        else:
            print(f"File already exists for {paper_id}: {pdf_file.name}")

        info = verify_pdf_identity(pdf_file, paper_id)
        print(f" Verified {paper_id}: {info['page_count']} pages, SHA256={info['sha256'][:16]}..., stamp_verified={info['stamp_verified']}")
        manifest_entries[paper_id] = info

    save_manifest(manifest_entries, manifest_path)
    print(f"Corpus manifest saved to {manifest_path} with {len(manifest_entries)} verified papers.")


if __name__ == "__main__":
    main()

