"""Typed error hierarchy for The Relay inference and RAG pipeline.

Every custom exception provides:
- A standardized machine-readable error code.
- Explicit retryable flag for the retry controller.
- Mapping to the 4 mandatory CSV status values: 'ok', 'rate_limited', 'timeout', 'error'.
"""

from typing import Literal, Optional

RequestStatusType = Literal["ok", "rate_limited", "timeout", "error"]


class RelayError(Exception):
    """Base exception for all Relay pipeline errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "RELAY_INTERNAL_ERROR",
        retryable: bool = False,
        status_mapping: RequestStatusType = "error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable
        self.status_mapping = status_mapping

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code='{self.error_code}', "
            f"retryable={self.retryable}, status='{self.status_mapping}', "
            f"msg='{self.message}')"
        )


# --- Client & Validation Errors ---


class ValidationError(RelayError):
    """Raised when incoming request payload fails schema or domain constraints."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            error_code="CLIENT_VALIDATION_ERROR",
            retryable=False,
            status_mapping="error",
        )


class RequestDeadlineExceededError(RelayError):
    """Raised when request execution exceeds the client-specified patience deadline."""

    def __init__(
        self,
        message: str = "Request exceeded total patience deadline",
        deadline_ms: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        self.deadline_ms = deadline_ms
        self.elapsed_ms = elapsed_ms
        super().__init__(
            message=message,
            error_code="DEADLINE_EXCEEDED",
            retryable=False,
            status_mapping="timeout",
        )


# --- Admission & Queuing Errors ---


class QueueFullError(RelayError):
    """Raised when the request queue reaches capacity ceiling and sheds arrivals."""

    def __init__(self, message: str = "Request queue capacity reached ceiling") -> None:
        super().__init__(
            message=message,
            error_code="QUEUE_CAPACITY_SHED",
            retryable=False,
            status_mapping="rate_limited",
        )


class QueueTimeoutError(RelayError):
    """Raised when a request waits in the FIFO queue longer than queue_timeout_ms."""

    def __init__(self, message: str = "Request timed out waiting in queue") -> None:
        super().__init__(
            message=message,
            error_code="QUEUE_TIMEOUT",
            retryable=False,
            status_mapping="timeout",
        )


class DeadlineShedError(RelayError):
    """Raised when proactive Little's Law check sheds request before enqueueing."""

    def __init__(self, message: str = "Shed proactively: estimated wait exceeds deadline") -> None:
        super().__init__(
            message=message,
            error_code="PROACTIVE_DEADLINE_SHED",
            retryable=False,
            status_mapping="timeout",
        )


class CircuitOpenError(RelayError):
    """Raised when the circuit breaker is open (e.g., daily budget exhausted or 5xx storm)."""

    def __init__(
        self,
        message: str = "Circuit breaker open; rejecting traffic",
        is_daily_quota: bool = False,
    ) -> None:
        self.is_daily_quota = is_daily_quota
        super().__init__(
            message=message,
            error_code="CIRCUIT_OPEN_DAILY_QUOTA" if is_daily_quota else "CIRCUIT_OPEN_FAULT",
            retryable=False,
            status_mapping="rate_limited" if is_daily_quota else "error",
        )


# --- Provider & Inference Errors ---


class ProviderError(RelayError):
    """Base error for external or stand-in LLM provider failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "PROVIDER_ERROR",
        retryable: bool = False,
        status_mapping: RequestStatusType = "error",
        status_code: Optional[int] = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(
            message=message,
            error_code=error_code,
            retryable=retryable,
            status_mapping=status_mapping,
        )


class ProviderUnavailableError(ProviderError):
    """Raised when provider returns HTTP 500, 502, 503, or 504."""

    def __init__(
        self,
        message: str = "Provider returned 5xx server error",
        status_code: int = 503,
    ) -> None:
        super().__init__(
            message=message,
            error_code="PROVIDER_UNAVAILABLE",
            retryable=True,
            status_mapping="error",
            status_code=status_code,
        )


class ProviderTimeoutError(ProviderError):
    """Raised when provider network call or SSE chunk wait exceeds stage timeout."""

    def __init__(self, message: str = "Provider socket or chunk wait timed out") -> None:
        super().__init__(
            message=message,
            error_code="PROVIDER_TIMEOUT",
            retryable=True,
            status_mapping="timeout",
        )


class ProviderAuthenticationError(ProviderError):
    """Raised when provider rejects API key (HTTP 401). Terminal."""

    def __init__(self, message: str = "Provider rejected API key (HTTP 401)") -> None:
        super().__init__(
            message=message,
            error_code="PROVIDER_AUTH_FAILED",
            retryable=False,
            status_mapping="error",
        )


class RateLimitError(ProviderError):
    """Raised when provider throttles with HTTP 429."""

    def __init__(
        self,
        message: str = "Provider rate limit exceeded (HTTP 429)",
        retry_after_s: Optional[float] = None,
        limit_type: Optional[str] = None,
        is_daily: bool = False,
    ) -> None:
        self.retry_after_s = retry_after_s
        self.limit_type = limit_type
        self.is_daily = is_daily
        # Daily quota exhaustion is strictly non-retryable; burst RPM/TPM is retryable if deadline permits
        retryable = not is_daily and (retry_after_s is not None and retry_after_s <= 4.0)
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_DAILY" if is_daily else "RATE_LIMIT_BURST",
            retryable=retryable,
            status_mapping="rate_limited",
        )


class DailyBudgetExhaustedError(RateLimitError):
    """Raised specifically when RPD or TPD daily limits are exhausted."""

    def __init__(
        self,
        message: str = "Daily free-tier request or token budget exhausted",
        reset_time_utc: Optional[str] = None,
    ) -> None:
        self.reset_time_utc = reset_time_utc
        super().__init__(
            message=message,
            retry_after_s=None,
            limit_type="RPD/TPD",
            is_daily=True,
        )


# --- RAG & Tool Execution Errors ---


class RetrievalError(RelayError):
    """Raised when in-memory retrieval fails or index is uninitialized."""

    def __init__(self, message: str = "Retrieval index search failed") -> None:
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            retryable=False,
            status_mapping="error",
        )


class ToolTimeoutError(RetrievalError):
    """Raised when Stage 2 retrieval tool execution exceeds tool timeout."""

    def __init__(self, message: str = "Stage 2 retrieval tool exceeded deadline") -> None:
        super().__init__(
            message=message,
        )
        self.error_code = "TOOL_TIMEOUT"
        self.status_mapping = "timeout"


class CitationVerificationError(RelayError):
    """Raised when Stage 4 detects an uncited claim, missing provenance, or invalid syntax."""

    def __init__(
        self,
        message: str,
        missing_claims: Optional[list[str]] = None,
        repair_attempted: bool = False,
    ) -> None:
        self.missing_claims = missing_claims or []
        self.repair_attempted = repair_attempted
        super().__init__(
            message=message,
            error_code="CITATION_VERIFICATION_FAILED",
            retryable=not repair_attempted,  # Allowed exactly 1 repair loop
            status_mapping="error",
        )


class EpistemicRefusalError(RelayError):
    """Not an error in pipeline execution, but a typed signal for intentional abstention.

    Mapped to status='ok' and answer_grade='refused_correctly' in CSV.
    """

    def __init__(
        self,
        message: str = "The research papers do not contain sufficient evidence to answer this question.",
        reason: str = "corpus_absent",
    ) -> None:
        self.reason = reason
        super().__init__(
            message=message,
            error_code="EPISTEMIC_REFUSAL",
            retryable=False,
            status_mapping="ok",
        )

