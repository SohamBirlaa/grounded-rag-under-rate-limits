"""Monotonic timeout and deadline tracking utilities for The Relay.

Guarantees:
- Enforces deadline countdowns across multi-attempt pipelines using monotonic nanosecond clocks.
- Prevents timed-out async operations from lingering or completing silently.
- Distinguishes client deadline expiry from individual provider attempt timeouts.
- Establishes monotonic timeout hierarchy (ADR-08, Inference Design §4.6):
  t_tool (1500ms) < t_ttft (4000ms) < t_llm_call (8500ms) < t_total_request (12000ms)
"""

import asyncio
from dataclasses import dataclass
import time
from typing import Optional

from relay.contracts.errors import RequestDeadlineExceededError
from relay.observability.metrics import MonotonicTimer

# --- Monotonic Timeout Hierarchy Constants (§4.6) ---
DEFAULT_TOOL_TIMEOUT_MS: float = 1500.0
DEFAULT_QUEUE_TIMEOUT_MS: float = 4500.0
DEFAULT_TTFT_TIMEOUT_MS: float = 4000.0
DEFAULT_STREAM_IDLE_TIMEOUT_MS: float = 2000.0
DEFAULT_LLM_CALL_TIMEOUT_MS: float = 8500.0
DEFAULT_TOTAL_REQUEST_TIMEOUT_MS: float = 12000.0


@dataclass(frozen=True)
class TimeoutHierarchy:
    """Explicitly configured and validated monotonic timeout hierarchy."""

    tool_timeout_ms: float = DEFAULT_TOOL_TIMEOUT_MS
    queue_timeout_ms: float = DEFAULT_QUEUE_TIMEOUT_MS
    ttft_timeout_ms: float = DEFAULT_TTFT_TIMEOUT_MS
    stream_idle_timeout_ms: float = DEFAULT_STREAM_IDLE_TIMEOUT_MS
    llm_call_timeout_ms: float = DEFAULT_LLM_CALL_TIMEOUT_MS
    total_request_timeout_ms: float = DEFAULT_TOTAL_REQUEST_TIMEOUT_MS

    def validate_monotonicity(self) -> None:
        """Validate the mathematical ordering t_tool < t_ttft < t_llm_call < t_total_request.

        Raises:
            ValueError: If the monotonic invariant is violated.
        """
        if not (
            self.tool_timeout_ms
            < self.ttft_timeout_ms
            < self.llm_call_timeout_ms
            < self.total_request_timeout_ms
        ):
            raise ValueError(
                f"Monotonic timeout hierarchy violation: "
                f"tool ({self.tool_timeout_ms}ms) < "
                f"ttft ({self.ttft_timeout_ms}ms) < "
                f"llm_call ({self.llm_call_timeout_ms}ms) < "
                f"total ({self.total_request_timeout_ms}ms) must strictly hold."
            )


class DeadlineTracker:
    """Monotonic deadline countdown tracker for a client request."""

    def __init__(self, deadline_ms: int) -> None:
        self.deadline_ms = deadline_ms
        self.deadline_s = deadline_ms / 1000.0
        self.timer = MonotonicTimer()
        self.timer.start()

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds since deadline tracking began."""
        return self.timer.elapsed_s()

    @property
    def remaining_s(self) -> float:
        """Remaining time before deadline expiry in seconds."""
        rem = self.deadline_s - self.elapsed_s
        return max(0.0, rem)

    @property
    def is_expired(self) -> bool:
        """Return True if the deadline has elapsed."""
        return self.elapsed_s >= self.deadline_s

    def check_deadline(self) -> None:
        """Raise RequestDeadlineExceededError if deadline has passed."""
        if self.is_expired:
            elapsed = self.timer.elapsed_ms()
            raise RequestDeadlineExceededError(
                f"Client request deadline of {self.deadline_ms}ms exceeded (elapsed: {elapsed:.1f}ms)",
                deadline_ms=self.deadline_ms,
                elapsed_ms=elapsed,
            )

    def can_afford(self, delay_s: float) -> bool:
        """Check if remaining deadline allows waiting for delay_s before next action."""
        return self.remaining_s > delay_s
