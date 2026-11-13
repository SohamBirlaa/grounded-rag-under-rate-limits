"""Reliability loop, retry controller, and error classification for The Relay.

Implements:
- Bounded exponential backoff with jitter and retry-after header parsing
- Transience classification preventing retries on auth, validation, or daily quota exhaustion
- Monotonic deadline-aware retry suppression
- Complete attempt, retry count, and delay tracking for 12-column CSV telemetry
- Local Retry Amplification Suppression (§4.5.2 & ADR-10):
  Suppresses retries when in-flight retries exceed 20% of completed requests in window
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
import random
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from relay.contracts.errors import (
    DailyBudgetExhaustedError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    RelayError,
    RequestDeadlineExceededError,
    ValidationError,
)
from relay.contracts.schemas import RequestStatusType
from relay.observability.logger import sanitize_message
from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
    ProviderStreamChunk,
)
from relay.reliability.timeouts import DeadlineTracker

logger = logging.getLogger("relay.reliability")


@dataclass
class RetryPolicy:
    """Configurable exponential backoff retry policy."""

    max_retries: int = 2
    base_backoff_s: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 4.0
    jitter: bool = True

    def compute_delay(self, attempt: int, retry_after_s: Optional[float] = None) -> float:
        """Calculate backoff delay for the next attempt."""
        if retry_after_s is not None and retry_after_s > 0:
            delay = min(self.max_backoff_s, retry_after_s)
        else:
            delay = min(
                self.max_backoff_s,
                self.base_backoff_s * (self.backoff_multiplier**attempt),
            )

        if self.jitter:
            delay += random.uniform(0.0, 0.1 * delay)

        return delay


class RetryAmplificationGuard:
    """Guards against retry storms by tracking the ratio of active retries to completed requests.

    Implements Inference Design §4.5.2:
    "If current in-flight retries exceed 20% of completed requests in the window,
    retries are suppressed to prevent cascading load."
    """

    def __init__(
        self,
        suppression_threshold_pct: float = 0.20,
        window_s: float = 60.0,
        min_sample_size: int = 5,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.suppression_threshold_pct = suppression_threshold_pct
        self.window_s = window_s
        self.min_sample_size = min_sample_size
        self._clock_fn = clock_fn or time.monotonic

        self._in_flight_retries: int = 0
        self._completion_timestamps: List[float] = []

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self._completion_timestamps = [t for t in self._completion_timestamps if t >= cutoff]

    @property
    def in_flight_retries(self) -> int:
        """Count of requests currently executing a retry attempt."""
        return self._in_flight_retries

    @property
    def window_completions(self) -> int:
        """Count of completed requests in the rolling measurement window."""
        self._prune(self._clock_fn())
        return len(self._completion_timestamps)

    def record_request_completion(self) -> None:
        """Record completion of a request to update baseline denominator."""
        now = self._clock_fn()
        self._completion_timestamps.append(now)
        self._prune(now)

    def is_suppressed(self) -> bool:
        """Check whether retry ratio exceeds the 20% suppression threshold."""
        now = self._clock_fn()
        self._prune(now)
        completions = len(self._completion_timestamps)

        if completions < self.min_sample_size:
            # Not enough samples to justify suppression
            return False

        ratio = self._in_flight_retries / completions
        if ratio > self.suppression_threshold_pct:
            logger.warning(
                "Retry amplification threshold exceeded: in_flight_retries=%d, window_completions=%d, ratio=%.2f > %.2f",
                self._in_flight_retries,
                completions,
                ratio,
                self.suppression_threshold_pct,
            )
            return True
        return False

    @asynccontextmanager
    async def track_retry(self) -> AsyncIterator[int]:
        """Context manager to track an active in-flight retry."""
        self._in_flight_retries += 1
        try:
            yield self._in_flight_retries
        finally:
            self._in_flight_retries = max(0, self._in_flight_retries - 1)


def classify_error_retryability(exc: Exception) -> Tuple[bool, str, RequestStatusType]:
    """Classify an exception into (is_retryable, reason, terminal_status_mapping)."""
    if isinstance(exc, DailyBudgetExhaustedError):
        return False, "Daily request or token quota exhausted", "rate_limited"

    if isinstance(exc, RateLimitError):
        if exc.is_daily:
            return False, "Daily rate limit exhausted", "rate_limited"
        # Burst RPM/TPM is retryable if retry-after is within budget
        if exc.retry_after_s is not None and exc.retry_after_s > 4.0:
            return False, f"Rate limit retry-after ({exc.retry_after_s}s) exceeds backoff ceiling", "rate_limited"
        return True, f"Burst rate limit: {exc.message}", "rate_limited"

    if isinstance(exc, ProviderAuthenticationError):
        return False, "Authentication failure (HTTP 401)", "error"

    if isinstance(exc, ValidationError):
        return False, "Client request validation failure", "error"

    if isinstance(exc, RequestDeadlineExceededError):
        return False, "Client deadline expired", "timeout"

    if isinstance(exc, (ProviderTimeoutError, asyncio.TimeoutError, TimeoutError)):
        return True, "Provider attempt timed out", "timeout"

    if isinstance(exc, ProviderUnavailableError):
        return True, f"Provider service unavailable: {exc.message}", "error"

    if isinstance(exc, RelayError):
        return exc.retryable, exc.message, exc.status_mapping

    # Unknown exception
    return False, f"Unhandled error: {str(exc)}", "error"


@dataclass
class ReliabilityResult:
    """Outcome of an inference invocation through the ReliabilityLoop."""

    success: bool
    chunks: List[ProviderStreamChunk]
    assembled_text: str
    first_token_ms: Optional[float]
    total_ms: float
    status: RequestStatusType
    retries: int
    attempts: int
    retry_reasons: List[str]
    tokens_in: int
    tokens_out: int
    first_token_arrival_s: Optional[float] = None
    final_error: Optional[Exception] = None


class ReliabilityLoop:
    """Reliability supervisor wrapping model provider invocations."""

    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        sleep_fn: Optional[Callable[[float], Any]] = None,
        amplification_guard: Optional[RetryAmplificationGuard] = None,
    ) -> None:
        self.policy = retry_policy or RetryPolicy()
        self._sleep_fn = sleep_fn or asyncio.sleep
        self.amplification_guard = amplification_guard

    async def execute_stream(
        self,
        provider: ModelProviderPort,
        request: ProviderCompletionRequest,
        deadline_tracker: Optional[DeadlineTracker] = None,
    ) -> ReliabilityResult:
        """Execute provider streaming with bounded retries and timeout supervision."""
        t_overall_start = time.perf_counter()
        attempts = 0
        retry_reasons: List[str] = []

        last_error: Optional[Exception] = None
        final_status: RequestStatusType = "error"

        try:
            while attempts <= self.policy.max_retries:
                attempts += 1
                chunks: List[ProviderStreamChunk] = []
                first_token_arrival_s: Optional[float] = None

                try:
                    # Check overall deadline before attempt
                    if deadline_tracker:
                        deadline_tracker.check_deadline()

                    # Calculate attempt timeout as min(request.timeout_s, remaining_deadline)
                    attempt_timeout = request.timeout_s
                    if deadline_tracker:
                        attempt_timeout = min(attempt_timeout, deadline_tracker.remaining_s)
                        if attempt_timeout <= 0.05:
                            raise RequestDeadlineExceededError(
                                "Insufficient remaining deadline to initiate attempt",
                                deadline_ms=deadline_tracker.deadline_ms,
                                elapsed_ms=deadline_tracker.timer.elapsed_ms(),
                            )

                    # Execute streaming within attempt timeout
                    async def _stream_collector() -> None:
                        nonlocal first_token_arrival_s
                        async for chunk in provider.stream_completion(request):
                            chunks.append(chunk)
                            if chunk.is_first_token and first_token_arrival_s is None:
                                first_token_arrival_s = chunk.arrival_monotonic_s

                    try:
                        await asyncio.wait_for(_stream_collector(), timeout=attempt_timeout)
                    except (asyncio.TimeoutError, TimeoutError) as timeout_exc:
                        if deadline_tracker and deadline_tracker.is_expired:
                            raise RequestDeadlineExceededError(
                                f"Overall request deadline of {deadline_tracker.deadline_ms}ms expired",
                                deadline_ms=deadline_tracker.deadline_ms,
                                elapsed_ms=deadline_tracker.timer.elapsed_ms(),
                            ) from timeout_exc
                        raise ProviderTimeoutError(
                            f"Provider attempt {attempts} timed out after {attempt_timeout:.2f}s"
                        ) from timeout_exc

                    # Success!
                    t_overall_end = time.perf_counter()
                    total_duration_ms = (t_overall_end - t_overall_start) * 1000.0

                    first_token_ms: Optional[float] = None
                    if first_token_arrival_s is not None:
                        first_token_ms = (first_token_arrival_s - t_overall_start) * 1000.0
                        first_token_ms = min(first_token_ms, total_duration_ms)

                    assembled_text = "".join(c.text_delta for c in chunks)

                    # Extract token usage from chunks if available
                    tokens_in = 0
                    tokens_out = 0
                    for c in reversed(chunks):
                        if c.prompt_tokens is not None and tokens_in == 0:
                            tokens_in = c.prompt_tokens
                        if c.completion_tokens is not None and tokens_out == 0:
                            tokens_out = c.completion_tokens

                    return ReliabilityResult(
                        success=True,
                        chunks=chunks,
                        assembled_text=assembled_text,
                        first_token_ms=first_token_ms,
                        first_token_arrival_s=first_token_arrival_s,
                        total_ms=total_duration_ms,
                        status="ok",
                        retries=attempts - 1,
                        attempts=attempts,
                        retry_reasons=retry_reasons,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        final_error=None,
                    )

                except asyncio.CancelledError:
                    logger.info("ReliabilityLoop execution cancelled externally on attempt %d", attempts)
                    raise

                except Exception as e:
                    last_error = e
                    is_retryable, reason, mapped_status = classify_error_retryability(e)
                    final_status = mapped_status
                    retry_reasons.append(reason)
                    logger.warning(
                        "Attempt %d/%d failed: %s (retryable=%s)",
                        attempts,
                        self.policy.max_retries + 1,
                        sanitize_message(str(e)),
                        is_retryable,
                    )

                    # If non-retryable or retries exhausted, break
                    if not is_retryable or attempts > self.policy.max_retries:
                        break

                    # Extract retry-after if RateLimitError
                    retry_after_s: Optional[float] = None
                    if isinstance(e, RateLimitError):
                        retry_after_s = e.retry_after_s

                    backoff_delay = self.policy.compute_delay(attempts - 1, retry_after_s)

                    # Check deadline before sleeping for backoff
                    if deadline_tracker and not deadline_tracker.can_afford(backoff_delay + 0.1):
                        logger.info("Local retry suppressed: remaining deadline insufficient for backoff")
                        final_status = "timeout"
                        break

                    # Check Retry Amplification Guard (§4.5.2 & ADR-10)
                    if self.amplification_guard and self.amplification_guard.is_suppressed():
                        logger.warning("Local retry suppressed: active retries exceed 20% of window completions")
                        final_status = "rate_limited"
                        retry_reasons.append("Retry amplification suppression triggered")
                        break

                    # Sleep before next attempt while tracking in-flight retry
                    if self.amplification_guard:
                        async with self.amplification_guard.track_retry():
                            await self._sleep_fn(backoff_delay)
                    else:
                        await self._sleep_fn(backoff_delay)

            # Retries exhausted or terminal error encountered
            total_duration_ms = (time.perf_counter() - t_overall_start) * 1000.0

            return ReliabilityResult(
                success=False,
                chunks=[],
                assembled_text="",
                first_token_ms=None,
                first_token_arrival_s=None,
                total_ms=total_duration_ms,
                status=final_status,
                retries=attempts - 1,
                attempts=attempts,
                retry_reasons=retry_reasons,
                tokens_in=0,
                tokens_out=0,
                final_error=last_error,
            )

        finally:
            if self.amplification_guard:
                self.amplification_guard.record_request_completion()
