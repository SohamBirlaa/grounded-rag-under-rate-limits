"""Unit tests for Phase 1 typed error hierarchy and status mapping."""

import pytest

from relay.contracts.errors import (
    CircuitOpenError,
    CitationVerificationError,
    DailyBudgetExhaustedError,
    DeadlineShedError,
    EpistemicRefusalError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QueueFullError,
    QueueTimeoutError,
    RateLimitError,
    RelayError,
    RequestDeadlineExceededError,
    RetrievalError,
    ToolTimeoutError,
    ValidationError,
)


def test_base_relay_error() -> None:
    """Verify base RelayError properties."""
    err = RelayError("Something went wrong", error_code="TEST_CODE", retryable=False, status_mapping="error")
    assert str(err) == "Something went wrong"
    assert err.error_code == "TEST_CODE"
    assert err.retryable is False
    assert err.status_mapping == "error"
    assert "RelayError" in repr(err)


def test_client_validation_error() -> None:
    """Verify validation error is non-retryable and maps to error."""
    err = ValidationError("Field missing")
    assert err.status_mapping == "error"
    assert err.retryable is False
    assert err.error_code == "CLIENT_VALIDATION_ERROR"


def test_queue_and_deadline_errors() -> None:
    """Verify queue shed and deadline errors map correctly."""
    # Capacity shed maps to rate_limited (user was turned away due to capacity)
    q_full = QueueFullError()
    assert q_full.status_mapping == "rate_limited"
    assert q_full.retryable is False
    assert q_full.error_code == "QUEUE_CAPACITY_SHED"

    # Queue timeout maps to timeout
    q_timeout = QueueTimeoutError()
    assert q_timeout.status_mapping == "timeout"
    assert q_timeout.retryable is False

    # Proactive deadline shed maps to timeout
    dl_shed = DeadlineShedError()
    assert dl_shed.status_mapping == "timeout"
    assert dl_shed.retryable is False


def test_circuit_open_error() -> None:
    """Verify circuit breaker error maps differently for daily quota vs. fault storm."""
    err_fault = CircuitOpenError(is_daily_quota=False)
    assert err_fault.status_mapping == "error"
    assert err_fault.error_code == "CIRCUIT_OPEN_FAULT"

    err_daily = CircuitOpenError(is_daily_quota=True)
    assert err_daily.status_mapping == "rate_limited"
    assert err_daily.error_code == "CIRCUIT_OPEN_DAILY_QUOTA"


def test_rate_limit_error_retryability() -> None:
    """Verify rate limit retryability: burst RPM/TPM is retryable if backoff is short; daily is not."""
    # Short retry-after is retryable
    burst_retryable = RateLimitError("Burst limit", retry_after_s=2.0, is_daily=False)
    assert burst_retryable.retryable is True
    assert burst_retryable.status_mapping == "rate_limited"
    assert burst_retryable.error_code == "RATE_LIMIT_BURST"

    # Long retry-after (>4s) is non-retryable to prevent retry amplification under load
    burst_non_retryable = RateLimitError("Burst limit", retry_after_s=10.0, is_daily=False)
    assert burst_non_retryable.retryable is False

    # Daily budget exhausted is strictly non-retryable
    daily_err = DailyBudgetExhaustedError("Daily budget gone", reset_time_utc="15:00 UTC")
    assert daily_err.retryable is False
    assert daily_err.is_daily is True
    assert daily_err.status_mapping == "rate_limited"
    assert daily_err.error_code == "RATE_LIMIT_DAILY"


def test_provider_errors() -> None:
    """Verify provider 5xx is retryable, while 401 auth is strictly non-retryable."""
    prov_5xx = ProviderUnavailableError("503 Service Unavailable")
    assert prov_5xx.retryable is True
    assert prov_5xx.status_mapping == "error"

    prov_timeout = ProviderTimeoutError("Read timed out")
    assert prov_timeout.retryable is True
    assert prov_timeout.status_mapping == "timeout"

    prov_auth = ProviderAuthenticationError("Invalid API key")
    assert prov_auth.retryable is False
    assert prov_auth.status_mapping == "error"


def test_epistemic_refusal_is_ok_status() -> None:
    """Verify that intentional epistemic refusal maps to status 'ok' (clean answer delivery)."""
    refusal = EpistemicRefusalError("Papers do not contain this information", reason="corpus_absent")
    assert refusal.status_mapping == "ok"
    assert refusal.retryable is False
    assert refusal.error_code == "EPISTEMIC_REFUSAL"
    assert refusal.reason == "corpus_absent"


def test_citation_verification_repair_loop() -> None:
    """Verify that citation verification error allows 1 repair attempt then becomes non-retryable."""
    first_failure = CitationVerificationError("Missing citation", repair_attempted=False)
    assert first_failure.retryable is True

    second_failure = CitationVerificationError("Missing citation after repair", repair_attempted=True)
    assert second_failure.retryable is False

