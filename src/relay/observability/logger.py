"""Structured JSONL trace logger and security-conscious audit emitter.

Guarantees:
- Zero API key or sensitive token leakage (regex and substring scrubbing).
- Monotonic UTC ISO-8601 timestamps.
- Thread-safe atomic file appending for traces and diagnostic spans.
"""

import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Optional

from relay.contracts.schemas import CSVRowRecord, ScenarioMetricsRecord, StageSpanRecord

# Regex pattern matching Groq API keys
GROQ_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9_-]{20,}")


def sanitize_message(text: str) -> str:
    """Scrub raw Groq API keys and authorization tokens from text."""
    if not isinstance(text, str):
        return text
    return GROQ_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize dictionary values to prevent credential leakage."""
    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        if "api_key" in k.lower() or "secret" in k.lower() or "auth" in k.lower():
            cleaned[k] = "[REDACTED]"
        elif isinstance(v, str):
            cleaned[k] = sanitize_message(v)
        elif isinstance(v, dict):
            cleaned[k] = sanitize_dict(v)
        elif isinstance(v, list):
            cleaned[k] = [sanitize_message(item) if isinstance(item, str) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned


class StructuredLogger:
    """Thread-safe structured logger emitting JSONL event lines."""

    def __init__(
        self,
        component_name: str,
        log_file_path: Optional[Path] = None,
        spans_file_path: Optional[Path] = None,
    ) -> None:
        self.component_name = component_name
        self.log_file_path = log_file_path
        self.spans_file_path = spans_file_path or Path("data/traces/stage_spans.jsonl")
        self._lock = threading.Lock()

    def _format_event(
        self,
        level: str,
        event: str,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        iso_now = (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            + f".{int(time.time() * 1000) % 1000:03d}Z"
        )
        record: Dict[str, Any] = {
            "timestamp": iso_now,
            "level": level.upper(),
            "component": self.component_name,
            "event": event,
        }
        if request_id:
            record["request_id"] = request_id
        if metadata:
            record["metadata"] = sanitize_dict(metadata)
        return record

    def log_event(
        self,
        level: str,
        event: str,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a structured event to console and optional log file."""
        record = self._format_event(level, event, request_id, metadata)
        line = json.dumps(record)

        # Output to console
        print(line)

        # Append to log file if configured
        if self.log_file_path:
            with self._lock:
                self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def info(self, event: str, request_id: Optional[str] = None, **kwargs: Any) -> None:
        self.log_event("INFO", event, request_id=request_id, metadata=kwargs)

    def warning(self, event: str, request_id: Optional[str] = None, **kwargs: Any) -> None:
        self.log_event("WARNING", event, request_id=request_id, metadata=kwargs)

    def error(self, event: str, request_id: Optional[str] = None, **kwargs: Any) -> None:
        self.log_event("ERROR", event, request_id=request_id, metadata=kwargs)

    def debug(self, event: str, request_id: Optional[str] = None, **kwargs: Any) -> None:
        self.log_event("DEBUG", event, request_id=request_id, metadata=kwargs)

    def log_stage_span(self, span: StageSpanRecord) -> None:
        """Atomically append a StageSpanRecord to data/traces/stage_spans.jsonl."""
        line = span.model_dump_json()
        with self._lock:
            self.spans_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.spans_file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def log_scenario_metrics(
        self,
        metrics: ScenarioMetricsRecord,
        metrics_file_path: Optional[Path] = None,
    ) -> None:
        """Write ScenarioMetricsRecord to data/traces/scenario_metrics.json."""
        target_path = metrics_file_path or Path("data/traces/scenario_metrics.json")
        data = metrics.model_dump()

        with self._lock:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Read existing records if file exists to maintain scenario dictionary
            existing: Dict[str, Any] = {}
            if target_path.is_file():
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            existing = loaded
                except Exception:
                    existing = {}

            scenario_key = f"{metrics.scenario}_{metrics.source}"
            existing[scenario_key] = data

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)

