"""Targeted unit tests for QueueAdmissionDispatcher and runtime queue admission.

Validates:
1. Runtime admission dispatcher signals a queued request.
2. FIFO ordering is preserved across contention and sequential releases.
3. Queue capacity shedding (QueueFullError) works when queue reaches ceiling.
4. Queue timeout and deadline eviction work without leaking slots.
5. Dispatcher startup and shutdown lifecycle operate with zero task leaks.
6. API /query endpoint admits and completes without hanging.
7. Concurrent requests strictly respect configured admission limits.
8. Fast-test mode correctly bypasses queue wait.
"""

import asyncio
from typing import List
import pytest
from fastapi.testclient import TestClient

from relay.api.app import create_app
from relay.config.settings import ReliabilityConfig
from relay.contracts.errors import QueueFullError, QueueTimeoutError
from relay.contracts.schemas import RelayRequest
from relay.engine.orchestrator import RelayEngine
from relay.observability.metrics import AtomicQueueGauge
from relay.reliability.concurrency import ConcurrencyGate
from relay.reliability.dispatcher import QueueAdmissionDispatcher
from relay.reliability.queue import BoundedRequestQueue
from relay.reliability.timeouts import DeadlineTracker


@pytest.mark.asyncio
async def test_dispatcher_signals_queued_request():
    """Verify dispatcher pops FIFO head and sets admitted_event when capacity is available."""
    config = ReliabilityConfig(max_in_flight_llm=2, max_queue_size=10)
    queue = BoundedRequestQueue(config=config)
    gate = ConcurrencyGate(config=config)
    dispatcher = QueueAdmissionDispatcher(queue=queue, concurrency_gate=gate)

    tracker = DeadlineTracker(deadline_ms=5000)
    item = await queue.admit_or_shed("req_1", tracker)

    assert queue.current_depth == 1
    assert not item.admitted_event.is_set()
    assert dispatcher.active_admissions == 0

    admitted_count = await dispatcher.pump_admissions()
    assert admitted_count == 1
    assert item.admitted_event.is_set()
    assert queue.current_depth == 0
    assert dispatcher.active_admissions == 1

    # Release slot
    await dispatcher.release_slot()
    assert dispatcher.active_admissions == 0


@pytest.mark.asyncio
async def test_dispatcher_fifo_ordering_preserved():
    """Verify requests are admitted in strict FIFO order under concurrency contention."""
    config = ReliabilityConfig(max_in_flight_llm=2, max_queue_size=10)
    queue = BoundedRequestQueue(config=config)
    gate = ConcurrencyGate(config=config)
    dispatcher = QueueAdmissionDispatcher(queue=queue, concurrency_gate=gate)

    items = []
    for i in range(4):
        t = DeadlineTracker(deadline_ms=10000)
        items.append(await queue.admit_or_shed(f"req_{i}", t))

    assert queue.current_depth == 4

    # Initial pump should admit exactly 2 requests (req_0, req_1)
    admitted = await dispatcher.pump_admissions()
    assert admitted == 2
    assert dispatcher.active_admissions == 2
    assert items[0].admitted_event.is_set()
    assert items[1].admitted_event.is_set()
    assert not items[2].admitted_event.is_set()
    assert not items[3].admitted_event.is_set()
    assert queue.current_depth == 2

    # Release slot for req_0 -> must admit req_2 (next in FIFO)
    await dispatcher.release_slot()
    assert dispatcher.active_admissions == 2
    assert items[2].admitted_event.is_set()
    assert not items[3].admitted_event.is_set()
    assert queue.current_depth == 1

    # Release slot for req_1 -> must admit req_3
    await dispatcher.release_slot()
    assert dispatcher.active_admissions == 2
    assert items[3].admitted_event.is_set()
    assert queue.current_depth == 0

    # Clean up remaining slots
    await dispatcher.release_slot()
    await dispatcher.release_slot()
    assert dispatcher.active_admissions == 0


@pytest.mark.asyncio
async def test_dispatcher_queue_capacity_shedding():
    """Verify QueueFullError is raised when queue capacity ceiling (3) is reached."""
    gauge = AtomicQueueGauge()
    config = ReliabilityConfig(max_queue_size=3)
    queue = BoundedRequestQueue(config=config, gauge=gauge, default_service_rate_req_per_s=100.0)

    for i in range(3):
        t = DeadlineTracker(deadline_ms=10000)
        await queue.admit_or_shed(f"req_{i}", t)

    assert queue.current_depth == 3

    with pytest.raises(QueueFullError):
        await queue.admit_or_shed("req_overflow", DeadlineTracker(deadline_ms=10000))

    assert gauge.total_shed == 1
    assert queue.current_depth == 3


@pytest.mark.asyncio
async def test_dispatcher_queue_timeout_and_eviction():
    """Verify queue timeout evicts waiting item without leaking admission slots."""
    virtual_time = 50.0

    def mock_clock() -> float:
        return virtual_time

    config = ReliabilityConfig(max_queue_size=5, queue_timeout_ms=300.0)
    queue = BoundedRequestQueue(
        config=config,
        clock_fn=mock_clock,
        default_service_rate_req_per_s=100.0,
    )
    gate = ConcurrencyGate(config=config)
    dispatcher = QueueAdmissionDispatcher(queue=queue, concurrency_gate=gate)

    tracker = DeadlineTracker(deadline_ms=5000)
    item = await queue.admit_or_shed("req_timeout", tracker)
    assert queue.current_depth == 1

    # Advance clock past 300ms timeout
    virtual_time += 0.350

    with pytest.raises(QueueTimeoutError):
        await queue.wait_for_admission(item)

    assert item.evicted is True
    assert queue.current_depth == 0

    # Pumping now must find nothing to admit and active_admissions remains 0
    admitted = await dispatcher.pump_admissions()
    assert admitted == 0
    assert dispatcher.active_admissions == 0


@pytest.mark.asyncio
async def test_dispatcher_lifecycle_zero_task_leaks():
    """Verify dispatcher supervisor task starts and stops cleanly with zero task leaks."""
    config = ReliabilityConfig(max_in_flight_llm=4, max_queue_size=10)
    queue = BoundedRequestQueue(config=config)
    gate = ConcurrencyGate(config=config)
    dispatcher = QueueAdmissionDispatcher(queue=queue, concurrency_gate=gate)

    assert not dispatcher.is_running

    await dispatcher.start()
    assert dispatcher.is_running

    # Multiple starts should be idempotent
    await dispatcher.start()
    assert dispatcher.is_running

    await dispatcher.stop()
    assert not dispatcher.is_running

    # Verify no dangling dispatcher task in event loop
    current_tasks = [t for t in asyncio.all_tasks() if t.get_name() == "relay_queue_admission_dispatcher"]
    assert len(current_tasks) == 0

    # Also test RelayEngine context manager lifecycle
    engine = RelayEngine(config=None, fast_test_mode=True)
    async with engine:
        assert engine.admission_dispatcher.is_running
    assert not engine.admission_dispatcher.is_running


@pytest.mark.asyncio
async def test_concurrent_requests_respect_admission_limits():
    """Verify concurrent requests strictly respect ConcurrencyGate.max_llm_concurrency."""
    concurrency_cap = 3
    config = ReliabilityConfig(max_in_flight_llm=concurrency_cap, max_queue_size=20)
    queue = BoundedRequestQueue(config=config, default_service_rate_req_per_s=100.0)
    gate = ConcurrencyGate(config=config)
    dispatcher = QueueAdmissionDispatcher(queue=queue, concurrency_gate=gate)

    peak_admissions = 0
    lock = asyncio.Lock()

    async def simulated_request(req_id: str) -> None:
        nonlocal peak_admissions
        tracker = DeadlineTracker(deadline_ms=5000)
        q_item = await queue.admit_or_shed(req_id, tracker)
        await dispatcher.pump_admissions()
        await queue.wait_for_admission(q_item)

        async with lock:
            if dispatcher.active_admissions > peak_admissions:
                peak_admissions = dispatcher.active_admissions
            assert dispatcher.active_admissions <= concurrency_cap

        # Simulate execution time
        await asyncio.sleep(0.02)
        await dispatcher.release_slot()

    # Launch 8 requests concurrently
    tasks = [asyncio.create_task(simulated_request(f"req_{i}")) for i in range(8)]
    await asyncio.gather(*tasks)

    assert peak_admissions == concurrency_cap
    assert dispatcher.active_admissions == 0
    assert queue.current_depth == 0


@pytest.mark.asyncio
async def test_fast_test_mode_bypasses_queue():
    """Verify fast_test_mode executes without entering queue or modifying queue depth."""
    config = ReliabilityConfig(max_in_flight_llm=4, max_queue_size=10)
    queue = BoundedRequestQueue(config=config)
    engine = RelayEngine(queue=queue, fast_test_mode=True)

    req = RelayRequest(
        request_id="req_fast_test",
        question="What is Scaled Dot-Product Attention in the Transformer?",
        scenario="steady",
        source="stand-in",
        retrieval_setting="S3_450_8to4_rrf_no",
        deadline_ms=5000,
    )

    response, csv_row = await engine.execute_request(req)
    assert response.status == "ok"
    assert queue.current_depth == 0
    assert queue.peak_depth == 0
    assert engine.admission_dispatcher.active_admissions == 0


def test_api_query_endpoint_admits_and_succeeds():
    """Verify FastAPI /query endpoint admits request and returns HTTP 200 without timeout."""
    engine = RelayEngine(fast_test_mode=False)
    app = create_app(engine=engine)

    with TestClient(app) as client:
        payload = {
            "request_id": "test_api_admit_ok",
            "question": "What is Scaled Dot-Product Attention in the Transformer?",
            "scenario": "steady",
            "source": "stand-in",
            "retrieval_setting": "S3_450_8to4_rrf_no",
            "deadline_ms": 10000,
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "test_api_admit_ok"
        assert data["status"] == "ok"
        assert not data["is_refusal"]
        assert data["latency_breakdown"]["waiting_ms"] >= 0.0

