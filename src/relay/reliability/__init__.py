"""Reliability, admission control, and fault tolerance for The Relay.

Provides:
- ConcurrencyGate: shared model and tool semaphores (ADR-04, §4.3.1)
- BoundedRequestQueue, QueuedRequest: FIFO queue with atomic gauge and proactive deadline shedding (ADR-08, §4.3)
- ClientRateLimiter: client-side RPM/TPM/RPD/TPD tracking with pre-reservation and header reconciliation (ADR-09, §4.4)
- CircuitBreaker, CircuitState: two-mode breaker with 5xx fault trip and 3:00 PM daily quota latch (ADR-11, §4.7)
- DeadlineTracker, TimeoutHierarchy: monotonic timeout hierarchy enforcement (ADR-08, §4.6)
- ReliabilityLoop, ReliabilityResult, RetryPolicy, RetryAmplificationGuard: retry controller with anti-amplification guard (ADR-10, §4.5)
- classify_error_retryability: exception transience categorization
"""

from relay.reliability.circuit_breaker import CircuitBreaker, CircuitState
from relay.reliability.concurrency import ConcurrencyGate
from relay.reliability.dispatcher import QueueAdmissionDispatcher
from relay.reliability.queue import BoundedRequestQueue, QueuedRequest
from relay.reliability.rate_limiter import ClientRateLimiter
from relay.reliability.retry import (
    ReliabilityLoop,
    ReliabilityResult,
    RetryAmplificationGuard,
    RetryPolicy,
    classify_error_retryability,
)
from relay.reliability.timeouts import (
    DEFAULT_LLM_CALL_TIMEOUT_MS,
    DEFAULT_QUEUE_TIMEOUT_MS,
    DEFAULT_STREAM_IDLE_TIMEOUT_MS,
    DEFAULT_TOOL_TIMEOUT_MS,
    DEFAULT_TOTAL_REQUEST_TIMEOUT_MS,
    DEFAULT_TTFT_TIMEOUT_MS,
    DeadlineTracker,
    TimeoutHierarchy,
)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "ConcurrencyGate",
    "QueueAdmissionDispatcher",
    "BoundedRequestQueue",
    "QueuedRequest",
    "ClientRateLimiter",
    "ReliabilityLoop",
    "ReliabilityResult",
    "RetryAmplificationGuard",
    "RetryPolicy",
    "classify_error_retryability",
    "DeadlineTracker",
    "TimeoutHierarchy",
    "DEFAULT_TOOL_TIMEOUT_MS",
    "DEFAULT_QUEUE_TIMEOUT_MS",
    "DEFAULT_TTFT_TIMEOUT_MS",
    "DEFAULT_STREAM_IDLE_TIMEOUT_MS",
    "DEFAULT_LLM_CALL_TIMEOUT_MS",
    "DEFAULT_TOTAL_REQUEST_TIMEOUT_MS",
]
