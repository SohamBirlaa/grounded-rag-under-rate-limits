"""Test doubles for unit tests ONLY. Nothing here is used to produce results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FakeChunk:
    chunk_id: str
    paper_id: str
    page: int
    section: str
    text: str
    content_type: str = "prose"
    version: str = "v1"
    token_count: int = 0
    structural_breadcrumb: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeywordIndex:
    """Ranks chunks by count of shared lowercase words (deterministic)."""
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k=5):
        qw = set(query.lower().split())
        scored = sorted(self.chunks, key=lambda c: (-len(qw & set(c.text.lower().split())), c.chunk_id))
        return [(c, float(len(qw & set(c.text.lower().split()))), i + 1) for i, c in enumerate(scored[:top_k])]

    def _get_model(self):
        return None
