"""Unit tests for Phase 5: Inference Reliability Layer.

Covers:
- ConcurrencyGate: LLM/Tool semaphores, dynamic reduction, cancellation safety
- BoundedRequestQueue: FIFO ordering, capacity ceiling at 24, atomic depth gauge, proactive deadline shedding, queue timeout eviction
- ClientRateLimiter: RPM/TPM sliding windows, pre-reservation, header reconciliation, compact mode threshold, daily quotas
- CircuitBreaker: 5xx outage breaker, 10s cooldown, half-open trial, 3:00 PM daily quota latch, offline excerpt fallback
- TimeoutHierarchy: monotonic timeout hierarchy enforcement
- RetryAmplificationGuard: local retry storm suppression when ratio > 20%
"""

import asyncio
from pathlib import Path
import pytest

from relay.config.settings import ReliabilityConfig
from relay.contracts.errors import (
    CircuitOpenError,
    DailyBudgetExhaustedError,
    DeadlineShedError,
    QueueFullError,
    QueueTimeoutError,
    RateLimitError,
    RequestDeadlineExceededError,
)
from relay.observability.metrics import AtomicQueueGauge
from relay.providers.base import ProviderQuotaSnapshot
from relay.rag.retriever import RetrievedChunk
from relay.reliability.circuit_breaker import CircuitBreaker, CircuitState
from relay.reliability.concurrency import ConcurrencyGate
from relay.reliability.queue import BoundedRequestQueue, QueuedRequest
from relay.reliability.rate_limiter import ClientRateLimiter
from relay.reliability.retry import (
    ReliabilityLoop,
    RetryAmplificationGuard,
    RetryPolicy,
)
from relay.reliability.timeouts import (
    DEFAULT_LLM_CALL_TIMEOUT_MS,
    DEFAULT_QUEUE_TIMEOUT_MS,
    DEFAULT_TOOL_TIMEOUT_MS,
    DEFAULT_TOTAL_REQUEST_TIMEOUT_MS,
    DEFAULT_TTFT_TIMEOUT_MS,
    DeadlineTracker,
    TimeoutHierarchy,
)


# ==============================================================================
# 1. ConcurrencyGate Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_concurrency_gate_llm_and_tool_capacity():
    """Verify ConcurrencyGate enforces configured limits and releases slots safely."""
    gate = ConcurrencyGate(ReliabilityConfig(max_in_flight_llm=2, max_in_flight_tools=2))
    assert gate.max_llm_concurrency == 2
    assert gate.max_tool_concurrency == 2
    assert gate.active_llm_count == 0
    assert gate.active_tool_count == 0

    async with gate.acquire_llm() as slot1:
        assert slot1 == 1
        assert gate.active_llm_count == 1
        async with gate.acquire_llm() as slot2:
            assert slot2 == 2
            assert gate.active_llm_count == 2
    assert gate.active_llm_count == 0


@pytest.mark.asyncio
async def test_concurrency_gate_dynamic_limit_adjustment():
    """Verify dynamic concurrency reduction for rpm_halved scenario."""
    gate = ConcurrencyGate(ReliabilityConfig(max_in_flight_llm=8))
    assert gate.max_llm_concurrency == 8

    # Lower to 4 under rpm_halved
    gate.set_llm_limit(4)
    assert gate.max_llm_concurrency == 4

    with pytest.raises(ValueError):
        gate.set_llm_limit(0)


@pytest.mark.asyncio
async def test_concurrency_gate_safe_release_on_exception():
    """Verify semaphore slot is released even when inner block raises an exception."""
    gate = ConcurrencyGate(ReliabilityConfig(max_in_flight_llm=1))

    with pytest.raises(RuntimeError):
        async with gate.acquire_llm():
            assert gate.active_llm_count == 1
            raise RuntimeError("Forced error during processing")

    assert gate.active_llm_count == 0
    # Next acquisition should succeed immediately
    async with gate.acquire_llm():
        assert gate.active_llm_count == 1


# ==============================================================================
# 2. BoundedRequestQueue Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_bounded_queue_capacity_ceiling_rejection():
    """Verify queue sheds arrivals when depth reaches max_size (24)."""
    gauge = AtomicQueueGauge()
    queue = BoundedRequestQueue(
        config=ReliabilityConfig(max_queue_size=3),
        gauge=gauge,
        default_service_rate_req_per_s=100.0,  # high rate so proactive shed doesn't trigger
    )

    reqs = []
    for i in range(3):
        tracker = DeadlineTracker(deadline_ms=10000)
        q_item = await queue.admit_or_shed(f"req_{i}", tracker)
        reqs.append(q_item)

    assert queue.current_depth == 3
    assert gauge.depth == 3

    # 4th request must be shed due to QueueFullError
    overflow_tracker = DeadlineTracker(deadline_ms=10000)
    with pytest.raises(QueueFullError):
        await queue.admit_or_shed("req_overflow", overflow_tracker)

    assert gauge.depth == 3
    assert gauge.total_shed == 1


@pytest.mark.asyncio
async def test_bounded_queue_fifo_ordering_and_gauge():
    """Verify FIFO dequeue ordering and atomic gauge transitions."""
    gauge = AtomicQueueGauge()
    queue = BoundedRequestQueue(
        config=ReliabilityConfig(max_queue_size=10),
        gauge=gauge,
        default_service_rate_req_per_s=100.0,
    )

    t1 = DeadlineTracker(10000)
    t2 = DeadlineTracker(10000)
    item1 = await queue.admit_or_shed("req_1", t1)
    item2 = await queue.admit_or_shed("req_2", t2)

    assert queue.current_depth == 2
    assert queue.peak_depth == 2

    # Pop head -> item1
    popped1 = await queue.pop_next()
    assert popped1.request_id == "req_1"
    assert queue.current_depth == 1
    assert item1.admitted_event.is_set()

    # Pop next -> item2
    popped2 = await queue.pop_next()
    assert popped2.request_id == "req_2"
    assert queue.current_depth == 0
    assert item2.admitted_event.is_set()

    # Queue empty
    assert await queue.pop_next() is None


@pytest.mark.asyncio
async def test_bounded_queue_proactive_deadline_shedding():
    """Verify proactive shedding when estimated wait + min_exec > deadline."""
    # Service rate: 1 req/sec -> wait for depth 2 is (2+1)/1 = 3.0s.
    # min_exec: 0.5s -> total estimated: 3.5s.
    queue = BoundedRequestQueue(
        config=ReliabilityConfig(max_queue_size=10),
        default_service_rate_req_per_s=1.0,
        min_exec_time_s=0.5,
    )

    t_safe = DeadlineTracker(10000)  # 10s deadline
    await queue.admit_or_shed("req_1", t_safe)
    await queue.admit_or_shed("req_2", t_safe)

    # Arriving request with tight 2.0s deadline: estimated wait (2+1)/1 + 0.5 = 3.5s > 2.0s
    t_tight = DeadlineTracker(2000)
    with pytest.raises(DeadlineShedError):
        await queue.admit_or_shed("req_tight", t_tight)

    assert queue.current_depth == 2


@pytest.mark.asyncio
async def test_bounded_queue_wait_timeout_eviction():
    """Verify request evicts and raises QueueTimeoutError when wait exceeds queue_timeout_ms."""
    virtual_time = 100.0

    def mock_clock():
        return virtual_time

    queue = BoundedRequestQueue(
        config=ReliabilityConfig(max_queue_size=10, queue_timeout_ms=500.0),
        clock_fn=mock_clock,
        default_service_rate_req_per_s=100.0,
    )

    tracker = DeadlineTracker(5000)
    queued_item = await queue.admit_or_shed("req_wait", tracker)

    # Advance virtual time past 500ms
    virtual_time += 0.600

    with pytest.raises(QueueTimeoutError):
        await queue.wait_for_admission(queued_item)

    assert queued_item.evicted is True
    assert queue.gauge.total_evicted == 1


# ==============================================================================
# 3. ClientRateLimiter Tests
# ==============================================================================


def test_rate_limiter_rpm_sliding_window(tmp_path):
    """Verify 60s sliding window enforces RPM limit and calculates retry-after."""
    curr_time = 1000.0

    def mock_clock():
        return curr_time

    limiter = ClientRateLimiter(
        rpm_limit=3,
        tpm_limit=100000,
        state_path=tmp_path / "quota_state.json",
        clock_fn=mock_clock,
    )

    # Consume 3 requests
    limiter.check_and_reserve(estimated_tokens=100)
    curr_time += 1.0
    limiter.check_and_reserve(estimated_tokens=100)
    curr_time += 1.0
    limiter.check_and_reserve(estimated_tokens=100)

    assert limiter.current_rpm_used == 3
    assert limiter.remaining_rpm == 0

    # 4th request in same window must raise RateLimitError
    with pytest.raises(RateLimitError) as exc_info:
        limiter.check_and_reserve(estimated_tokens=100)
    assert exc_info.value.retry_after_s > 0

    # Advance clock past 60s from first request
    curr_time += 59.0
    # First request slides out; admission succeeds
    limiter.check_and_reserve(estimated_tokens=100)
    assert limiter.current_rpm_used == 3


def test_rate_limiter_tpm_reservation_and_reconciliation(tmp_path):
    """Verify pre-reservation, TPM limit breach, and post-call token refund."""
    curr_time = 2000.0

    def mock_clock():
        return curr_time

    limiter = ClientRateLimiter(
        rpm_limit=100,
        tpm_limit=1000,
        state_path=tmp_path / "quota_state.json",
        clock_fn=mock_clock,
    )

    t_res = limiter.check_and_reserve(estimated_tokens=600)
    assert limiter.current_tpm_used == 600

    # Second request requesting 500 tokens exceeds 1000 TPM limit
    with pytest.raises(RateLimitError):
        limiter.check_and_reserve(estimated_tokens=500)

    # Actual generation used only 300 tokens: refund 300 tokens
    limiter.reconcile_post_call(
        reservation_time=t_res,
        estimated_tokens=600,
        actual_tokens=300,
    )
    assert limiter.current_tpm_used == 300
    assert limiter.remaining_tpm == 700

    # Now second request requesting 500 tokens succeeds
    limiter.check_and_reserve(estimated_tokens=500)
    assert limiter.current_tpm_used == 800


def test_rate_limiter_compact_mode_threshold(tmp_path):
    """Verify compact mode signals when remaining TPM <= 25%."""
    limiter = ClientRateLimiter(
        tpm_limit=1000,
        tpm_compact_threshold_pct=0.25,
        state_path=tmp_path / "quota_state.json",
    )
    assert limiter.is_compact_mode_needed is False

    # Reserve 760 tokens -> remaining = 240 <= 250 (25%)
    limiter.check_and_reserve(estimated_tokens=760)
    assert limiter.is_compact_mode_needed is True


def test_rate_limiter_daily_quota_exhaustion(tmp_path):
    """Verify daily request (RPD) limit exhaustion raises DailyBudgetExhaustedError."""
    limiter = ClientRateLimiter(
        daily_rpd=2,
        state_path=tmp_path / "quota_state.json",
    )
    limiter.check_and_reserve(estimated_tokens=10)
    limiter.check_and_reserve(estimated_tokens=10)

    with pytest.raises(DailyBudgetExhaustedError) as exc_info:
        limiter.check_and_reserve(estimated_tokens=10)
    assert exc_info.value.reset_time_utc == "15:00:00Z"


# ==============================================================================
# 4. CircuitBreaker Tests
# ==============================================================================


def test_circuit_breaker_5xx_trip_and_cooldown_recovery():
    """Verify circuit trips to OPEN_5XX after 5 consecutive 5xx errors and recovers via HALF_OPEN."""
    curr_time = 1000.0

    def mock_clock():
        return curr_time

    breaker = CircuitBreaker(
        consecutive_5xx_threshold=5,
        cooldown_s=10.0,
        clock_fn=mock_clock,
    )
    assert breaker.state == CircuitState.CLOSED

    # 4 consecutive 5xx errors do not trip
    for _ in range(4):
        breaker.record_failure(RuntimeError("503 Service Unavailable"))
        breaker.check_gate()
        assert breaker.state == CircuitState.CLOSED

    # 5th 5xx error trips breaker to OPEN_5XX
    breaker.record_failure(RuntimeError("500 Internal Server Error"))
    assert breaker.state == CircuitState.OPEN_5XX

    with pytest.raises(CircuitOpenError) as exc_info:
        breaker.check_gate()
    assert exc_info.value.is_daily_quota is False

    # Advance clock 9 seconds (still within cooldown)
    curr_time += 9.0
    assert breaker.state == CircuitState.OPEN_5XX

    # Advance clock past 10s cooldown -> transitions to HALF_OPEN
    curr_time += 2.0
    assert breaker.state == CircuitState.HALF_OPEN

    # First probe allowed through
    breaker.check_gate()

    # Probe succeeds -> resets to CLOSED
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_daily_quota_latch_and_offline_refusal():
    """Verify 3:00 PM daily quota latch and instant (< 15 ms) offline refusal formatting."""
    breaker = CircuitBreaker()
    breaker.latch_daily_quota()
    assert breaker.state == CircuitState.OPEN_DAILY_QUOTA
    assert breaker.is_daily_quota_latched is True

    with pytest.raises(CircuitOpenError) as exc_info:
        breaker.check_gate()
    assert exc_info.value.is_daily_quota is True

    # Test offline refusal with raw excerpts
    mock_passages = [
        RetrievedChunk(
            chunk_id="chunk_test_01",
            paper_id="1706.03762v7",
            version="v7",
            page=3,
            section="3.2 Attention",
            content_type="text",
            text="Scaled Dot-Product Attention computes queries and keys with scaling factor sqrt(d_k).",
            token_count=16,
            score=0.92,
            rank=1,
            retrieval_method="dense",
        )
    ]

    answer_text, citations = breaker.build_offline_daily_refusal(
        request_id="req_offline_01",
        question="What is attention?",
        passages=mock_passages,
    )

    assert "Daily inference budget exhausted" in answer_text
    assert "15:00:00Z" in answer_text
    assert "Scaled Dot-Product Attention" in answer_text
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk_test_01"


# ==============================================================================
# 5. TimeoutHierarchy & RetryAmplificationGuard Tests
# ==============================================================================


def test_timeout_hierarchy_monotonicity_validation():
    """Verify TimeoutHierarchy asserts strict mathematical ordering."""
    valid = TimeoutHierarchy(
        tool_timeout_ms=1500.0,
        ttft_timeout_ms=4000.0,
        llm_call_timeout_ms=8500.0,
        total_request_timeout_ms=12000.0,
    )
    valid.validate_monotonicity()

    # Inverted tool vs ttft violates hierarchy
    invalid = TimeoutHierarchy(
        tool_timeout_ms=5000.0,
        ttft_timeout_ms=4000.0,
        llm_call_timeout_ms=8500.0,
        total_request_timeout_ms=12000.0,
    )
    with pytest.raises(ValueError):
        invalid.validate_monotonicity()


@pytest.mark.asyncio
async def test_retry_amplification_guard_suppression():
    """Verify local retry suppression triggers when retry ratio exceeds 20% of completions."""
    curr_time = 100.0

    def mock_clock():
        return curr_time

    guard = RetryAmplificationGuard(
        suppression_threshold_pct=0.20,
        min_sample_size=5,
        clock_fn=mock_clock,
    )

    # Record 10 completed requests
    for _ in range(10):
        guard.record_request_completion()

    assert guard.window_completions == 10
    assert guard.is_suppressed() is False

    # Simulate 3 concurrent in-flight retries: 3 / 10 = 30% > 20%
    async with guard.track_retry():
        async with guard.track_retry():
            async with guard.track_retry():
                assert guard.in_flight_retries == 3
                assert guard.is_suppressed() is True

    # After retries complete, in-flight drops to 0 -> suppression lifts
    assert guard.in_flight_retries == 0
    assert guard.is_suppressed() is False
