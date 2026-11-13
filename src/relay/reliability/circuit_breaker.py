"""Two-mode circuit breaker and 3:00 PM daily quota latch for The Relay.

Implements ADR-11 and Inference Design Section 4.7:
- States: CLOSED, OPEN_5XX, HALF_OPEN, OPEN_DAILY_QUOTA
- Mode 1: Provider 5xx Outage Breaker:
    - Trips to OPEN_5XX after 5 consecutive qualifying 5xx failures.
    - 10.0s cooldown before transitioning to HALF_OPEN probe state.
    - Single trial probe determines recovery to CLOSED or return to OPEN_5XX.
- Mode 2: 3:00 PM Daily Quota Latch:
    - Latches OPEN_DAILY_QUOTA when RPD == 0, TPD < 400, or DailyBudgetExhaustedError.
    - Immediate (< 15 ms) rejection without queuing or calling Groq.
    - Offline fallback providing top-3 raw paper excerpts and reset timestamp (15:00:00Z).
- Fully supports virtual clocks for deterministic sub-second testing.
"""

from enum import Enum
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from relay.contracts.errors import CircuitOpenError, DailyBudgetExhaustedError
from relay.contracts.schemas import CitationRecord, RelayResponse
from relay.rag.retriever import RetrievedChunk

logger = logging.getLogger("relay.reliability.circuit_breaker")


class CircuitState(str, Enum):
    """Circuit breaker operational states."""

    CLOSED = "CLOSED"
    OPEN_5XX = "OPEN_5XX"
    HALF_OPEN = "HALF_OPEN"
    OPEN_DAILY_QUOTA = "OPEN_DAILY_QUOTA"


class CircuitBreaker:
    """Two-mode circuit breaker with transient fault protection and daily quota latch."""

    def __init__(
        self,
        consecutive_5xx_threshold: int = 5,
        cooldown_s: float = 10.0,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.consecutive_5xx_threshold = consecutive_5xx_threshold
        self.cooldown_s = cooldown_s
        self._clock_fn = clock_fn or time.monotonic

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_5xx_count: int = 0
        self._trip_timestamp: float = 0.0
        self._half_open_in_flight: bool = False

    @property
    def state(self) -> CircuitState:
        """Current circuit state with automatic cooldown evaluation."""
        if self._state == CircuitState.OPEN_5XX:
            now = self._clock_fn()
            if now - self._trip_timestamp >= self.cooldown_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_in_flight = False
                logger.info("Circuit breaker cooldown expired; transitioned to HALF_OPEN")
        return self._state

    @property
    def is_closed(self) -> bool:
        """Return True if circuit is allowing normal traffic."""
        return self.state == CircuitState.CLOSED

    @property
    def is_daily_quota_latched(self) -> bool:
        """Return True if daily budget latch is active."""
        return self._state == CircuitState.OPEN_DAILY_QUOTA

    def check_gate(self) -> None:
        """Check whether incoming traffic is permitted to execute.

        Raises:
            CircuitOpenError: If breaker is OPEN_5XX, HALF_OPEN (probe busy), or OPEN_DAILY_QUOTA.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN_DAILY_QUOTA:
            raise CircuitOpenError(
                message="Daily inference quota is exhausted. Circuit is latched until 15:00:00 UTC.",
                is_daily_quota=True,
            )

        if current_state == CircuitState.OPEN_5XX:
            remaining_cooldown = max(0.1, round(self.cooldown_s - (self._clock_fn() - self._trip_timestamp), 2))
            raise CircuitOpenError(
                message=f"Provider 5xx fault storm detected. Circuit breaker open for {remaining_cooldown}s",
                is_daily_quota=False,
            )

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_in_flight:
                raise CircuitOpenError(
                    message="Circuit is HALF_OPEN and trial request is already in progress.",
                    is_daily_quota=False,
                )
            # Mark probe in flight
            self._half_open_in_flight = True

    def record_success(self) -> None:
        """Record a successful provider execution."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("HALF_OPEN trial request succeeded; resetting circuit to CLOSED")
            self._state = CircuitState.CLOSED
            self._half_open_in_flight = False

        self._consecutive_5xx_count = 0

    def record_failure(self, error: Exception) -> None:
        """Record an error and evaluate circuit state transition."""
        if isinstance(error, DailyBudgetExhaustedError):
            self.latch_daily_quota()
            return

        is_5xx = False
        error_msg = str(error).lower()
        if any(code in error_msg for code in ["500", "502", "503", "504", "server_error"]):
            is_5xx = True

        if is_5xx:
            self._consecutive_5xx_count += 1
            logger.warning("Provider 5xx error recorded (%d consecutive)", self._consecutive_5xx_count)

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Trial probe failed in HALF_OPEN; reopening breaker to OPEN_5XX")
                self._state = CircuitState.OPEN_5XX
                self._trip_timestamp = self._clock_fn()
                self._half_open_in_flight = False
            elif self._consecutive_5xx_count >= self.consecutive_5xx_threshold:
                logger.error(
                    "Threshold reached (%d consecutive 5xx); tripping circuit breaker to OPEN_5XX",
                    self._consecutive_5xx_count,
                )
                self._state = CircuitState.OPEN_5XX
                self._trip_timestamp = self._clock_fn()
        else:
            # Client error, timeout, or non-5xx error: resets consecutive 5xx counter
            self._consecutive_5xx_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_in_flight = False

    def latch_daily_quota(self) -> None:
        """Latch the circuit breaker permanently for the day until 15:00 UTC."""
        self._state = CircuitState.OPEN_DAILY_QUOTA
        logger.warning("Circuit breaker latched: OPEN_DAILY_QUOTA until 15:00 UTC")

    def reset(self) -> None:
        """Reset breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_5xx_count = 0
        self._trip_timestamp = 0.0
        self._half_open_in_flight = False

    def build_offline_daily_refusal(
        self,
        request_id: str,
        question: str,
        passages: Sequence[RetrievedChunk],
        reset_time_utc: str = "15:00:00Z",
    ) -> Tuple[str, List[CitationRecord]]:
        """Construct instant (< 15 ms) offline refusal answer text with top raw excerpts.

        Implements Heading 7 requirement:
        Refuses to guess without verification, provides reset timestamp,
        and presents top retrieved raw paper excerpts with citations.
        """
        excerpts_text_parts: List[str] = []
        citations: List[CitationRecord] = []

        excerpts_text_parts.append(
            f"Daily inference budget exhausted. Service resumes at {reset_time_utc}. "
            "In accordance with our epistemic verification policy, the system refuses to answer from "
            "unverified parametric memory without running the verification model.\n\n"
            "Below are the top relevant raw paper excerpts retrieved deterministically from the corpus:"
        )

        # Include up to top 3 passages
        for idx, p in enumerate(passages[:3], start=1):
            excerpt = (
                f"\n\n--- Excerpt {idx} [arXiv:{p.paper_id}{p.version} p.{p.page} §{p.section}] ---\n"
                f"{p.text.strip()}"
            )
            excerpts_text_parts.append(excerpt)
            cit = CitationRecord(
                paper_id=p.paper_id,
                version=p.version,
                section=p.section,
                page=p.page,
                chunk_id=p.chunk_id,
                quoted_evidence=p.text[:60],
            )
            citations.append(cit)

        answer_text = "".join(excerpts_text_parts)
        return answer_text, citations
