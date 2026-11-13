"""Unit tests for the ReliabilityLoop, retry policy, error classification, and cancellation."""

import asyncio
from typing import AsyncIterator, List, Optional
import pytest

from relay.contracts.errors import (
    DailyBudgetExhaustedError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
    RequestDeadlineExceededError,
    ValidationError,
)
from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
    ProviderQuotaSnapshot,
    ProviderStreamChunk,
)
from relay.providers.standin import StandInProviderAdapter
from relay.reliability.retry import (
    ReliabilityLoop,
    RetryPolicy,
    classify_error_retryability,
)
from relay.reliability.timeouts import DeadlineTracker


class MockTimeoutProvider:
    """Mock provider with controllable timeouts and stream yields."""

    def __init__(self, timeout_attempts: int = 1) -> None:
        self.timeout_attempts = timeout_attempts
        self.calls = 0

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        return ProviderQuotaSnapshot()

    async def check_health(self) -> bool:
        return True

    async def stream_completion(
        self, request: ProviderCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        self.calls += 1
        if self.calls <= self.timeout_attempts:
            raise ProviderTimeoutError(f"Mock attempt {self.calls} timed out")

        yield ProviderStreamChunk(
            text_delta="Success after timeout",
            is_first_token=True,
            token_index=1,
            arrival_monotonic_s=100.0,
            prompt_tokens=10,
            completion_tokens=5,
        )


class MockCancelledProvider:
    """Mock provider simulating external task cancellation."""

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        return ProviderQuotaSnapshot()

    async def check_health(self) -> bool:
        return True

    async def stream_completion(
        self, request: ProviderCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        raise asyncio.CancelledError("External task cancellation")
        yield  # make it a generator


def test_classify_error_retryability():
    """Verify correct retry classification across all error categories."""
    # Non-retryable
    assert classify_error_retryability(DailyBudgetExhaustedError())[0] is False
    assert classify_error_retryability(ProviderAuthenticationError())[0] is False
    assert classify_error_retryability(ValidationError("Bad request"))[0] is False
    assert classify_error_retryability(RequestDeadlineExceededError("Expired"))[0] is False

    # Retryable
    assert classify_error_retryability(RateLimitError("Burst limit", retry_after_s=1.5))[0] is True
    assert classify_error_retryability(ProviderUnavailableError("503 error"))[0] is True
    assert classify_error_retryability(ProviderTimeoutError("Connect timeout"))[0] is True
    assert classify_error_retryability(asyncio.TimeoutError())[0] is True
    assert classify_error_retryability(TimeoutError())[0] is True


def test_retry_policy_delay_calculation():
    """Verify exponential backoff calculation and retry-after header priority."""
    policy = RetryPolicy(base_backoff_s=0.5, backoff_multiplier=2.0, max_backoff_s=4.0, jitter=False)

    # Attempt 0: 0.5s
    assert policy.compute_delay(attempt=0) == 0.5
    # Attempt 1: 1.0s
    assert policy.compute_delay(attempt=1) == 1.0
    # Attempt 2: 2.0s
    assert policy.compute_delay(attempt=2) == 2.0
    # Attempt 3: capped at max_backoff_s (4.0s)
    assert policy.compute_delay(attempt=3) == 4.0

    # Explicit retry-after
    assert policy.compute_delay(attempt=0, retry_after_s=1.8) == 1.8


def test_reliability_loop_immediate_success():
    """Verify ReliabilityLoop executes single attempt on immediate success."""
    async def _run():
        standin = StandInProviderAdapter(inject_delay=False)
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        result = await loop.execute_stream(standin, req)

        assert result.success is True
        assert result.status == "ok"
        assert result.attempts == 1
        assert result.retries == 0
        assert result.first_token_ms is not None
        assert result.first_token_ms <= result.total_ms
        assert result.first_token_arrival_s is not None
        assert len(result.assembled_text) > 0

    asyncio.run(_run())


def test_reliability_loop_transient_failure_then_success():
    """Verify ReliabilityLoop retries on transient error and succeeds."""
    async def _run():
        standin = StandInProviderAdapter(
            inject_delay=False,
            failure_mode="server_error",
            fail_attempts=1,  # fails attempt 1, succeeds attempt 2
        )
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        result = await loop.execute_stream(standin, req)

        assert result.success is True
        assert result.status == "ok"
        assert result.attempts == 2
        assert result.retries == 1
        assert len(result.retry_reasons) == 1
        assert "Provider service unavailable" in result.retry_reasons[0]

    asyncio.run(_run())


def test_reliability_loop_retries_exhausted():
    """Verify ReliabilityLoop terminates when max_retries is reached."""
    async def _run():
        standin = StandInProviderAdapter(
            inject_delay=False,
            failure_mode="server_error",
            fail_attempts=10,  # always fails
        )
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        result = await loop.execute_stream(standin, req)

        assert result.success is False
        assert result.status == "error"
        assert result.attempts == 3  # Initial + 2 retries
        assert result.retries == 2
        assert result.first_token_ms is None

    asyncio.run(_run())


def test_reliability_loop_attempt_timeout_retry_and_succeed():
    """Verify transient attempt timeout is retried and succeeds on attempt 2."""
    async def _run():
        provider = MockTimeoutProvider(timeout_attempts=1)
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        result = await loop.execute_stream(provider, req)

        assert result.success is True
        assert result.status == "ok"
        assert result.attempts == 2
        assert result.retries == 1
        assert "Success after timeout" in result.assembled_text

    asyncio.run(_run())


def test_reliability_loop_repeated_timeouts_exhaust_retries():
    """Verify repeated attempt timeouts exhaust retries with status='timeout'."""
    async def _run():
        provider = MockTimeoutProvider(timeout_attempts=5)
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        result = await loop.execute_stream(provider, req)

        assert result.success is False
        assert result.status == "timeout"
        assert result.attempts == 3  # Initial + 2 retries
        assert result.retries == 2
        assert result.first_token_ms is None

    asyncio.run(_run())


def test_reliability_loop_cancellation_propagation():
    """Verify external task cancellation propagates without being converted to error or retried."""
    async def _run():
        provider = MockCancelledProvider()
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        with pytest.raises(asyncio.CancelledError):
            await loop.execute_stream(provider, req)

    asyncio.run(_run())


def test_reliability_loop_non_retryable_auth_failure():
    """Verify non-retryable error terminates immediately on first attempt."""
    async def _run():
        standin = StandInProviderAdapter(
            inject_delay=False,
            failure_mode="auth_error",
        )
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        loop = ReliabilityLoop(retry_policy=RetryPolicy(max_retries=2), sleep_fn=noop_sleep)
        result = await loop.execute_stream(standin, req)

        assert result.success is False
        assert result.status == "error"
        assert result.attempts == 1
        assert result.retries == 0

    asyncio.run(_run())


def test_reliability_loop_deadline_suppression():
    """Verify retry is suppressed when remaining deadline is insufficient."""
    async def _run():
        standin = StandInProviderAdapter(
            inject_delay=False,
            failure_mode="server_error",
            fail_attempts=5,
        )
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        async def noop_sleep(_): pass

        # Set deadline of 50ms (less than backoff delay of 500ms)
        tracker = DeadlineTracker(deadline_ms=50)
        loop = ReliabilityLoop(
            retry_policy=RetryPolicy(max_retries=2, base_backoff_s=0.5),
            sleep_fn=noop_sleep,
        )
        result = await loop.execute_stream(standin, req, deadline_tracker=tracker)

        assert result.success is False
        assert result.status == "timeout"
        assert result.retries == 0  # Retry was suppressed!

    asyncio.run(_run())
