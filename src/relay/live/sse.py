"""Pure parsing helpers for Groq's OpenAI-compatible SSE stream.

Groq streams `data: {...}` lines. The final chunk carries usage either in a
top-level `usage` object or in `x_groq.usage`. Groq's usage also includes
server-side timing (`queue_time`, `prompt_time`, `completion_time`), which we
keep because it splits latency into provider-side waiting vs working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

DONE = "[DONE]"


@dataclass
class StreamDelta:
    content: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[str]:
    """Yield the payload of each `data:` line. Ignores comments/keep-alives."""
    for line in lines:
        if line is None:
            continue
        line = line.strip("\r\n")
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            yield line[5:].strip()


def normalize_usage(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract usage from a chunk (top-level `usage` or `x_groq.usage`)."""
    usage = obj.get("usage")
    if not usage:
        usage = (obj.get("x_groq") or {}).get("usage")
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or {}
    out = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
    }
    # Groq server-side timings in seconds -> keep as ms
    for key in ("queue_time", "prompt_time", "completion_time", "total_time"):
        if usage.get(key) is not None:
            out[f"{key}_ms"] = float(usage[key]) * 1000.0
    ctd = usage.get("completion_tokens_details") or {}
    if ctd.get("reasoning_tokens") is not None:
        out["reasoning_tokens"] = int(ctd["reasoning_tokens"])
    return out


def parse_payload(payload: str) -> Optional[StreamDelta]:
    """Parse one SSE payload. Returns None for the [DONE] sentinel."""
    if payload == DONE:
        return None
    obj = json.loads(payload)
    delta = StreamDelta(raw=obj, usage=normalize_usage(obj))
    choices = obj.get("choices") or []
    if choices:
        ch = choices[0]
        d = ch.get("delta") or {}
        delta.content = d.get("content") or ""
        # gpt-oss on Groq streams reasoning separately; never shown to users
        delta.reasoning = d.get("reasoning") or d.get("reasoning_content") or ""
        delta.finish_reason = ch.get("finish_reason")
    return delta
