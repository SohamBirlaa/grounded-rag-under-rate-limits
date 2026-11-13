"""results.csv writer (brief's columns first, extra columns appended) + JSONL traces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

BRIEF_COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status",
                 "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
# Appended so that peak queue, caching and model mix are recomputable from the same file.
EXTRA_COLUMNS = ["model", "path", "queue_wait_ms", "cached_tokens", "provider_first_token_ms"]


def _r(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.1f}"


class ResultsWriter:
    def __init__(self, csv_path: Path, trace_path: Path):
        self.csv_path, self.trace_path = Path(csv_path), Path(trace_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        self._f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=BRIEF_COLUMNS + EXTRA_COLUMNS)
        if new:
            self._w.writeheader()
        self._t = open(self.trace_path, "a", encoding="utf-8")

    def done_ids(self) -> Set[str]:
        if not self.csv_path.exists():
            return set()
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            return {row["request_id"] for row in csv.DictReader(f)}

    def write(self, rec: Any, grade: str = "", extra_trace: Optional[Dict[str, Any]] = None) -> None:
        self._w.writerow({
            "request_id": rec.request_id, "scenario": rec.scenario, "source": rec.source,
            "start_time": rec.start_time, "first_token_ms": _r(rec.first_token_ms), "total_ms": _r(rec.total_ms),
            "status": rec.status, "retries": rec.retries, "tokens_in": rec.tokens_in, "tokens_out": rec.tokens_out,
            "retrieval_setting": rec.setting_label, "answer_grade": grade,
            "model": rec.model, "path": rec.path, "queue_wait_ms": _r(rec.queue_wait_ms),
            "cached_tokens": rec.cached_tokens, "provider_first_token_ms": _r(rec.provider_first_token_ms),
        })
        self._f.flush()
        t = rec.to_trace()
        t["answer_grade"] = grade
        if extra_trace:
            t.update(extra_trace)
        self._t.write(json.dumps(t, default=str) + "\n")
        self._t.flush()

    def close(self) -> None:
        self._f.close()
        self._t.close()
