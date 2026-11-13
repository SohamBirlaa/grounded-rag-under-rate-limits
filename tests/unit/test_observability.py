"""Unit tests for Phase 1 observability: timers, queue gauges, latency breakdown, and logger."""

import json
from pathlib import Path
import time
import pytest

from relay.observability.logger import (
    StructuredLogger,
    sanitize_dict,
    sanitize_message,
)
from relay.observability.metrics import (
    AtomicQueueGauge,
    MonotonicTimer,
    StageLatencyTracker,
)


def test_monotonic_timer() -> None:
    """Verify MonotonicTimer measures elapsed ms and s using perf_counter_ns."""
    timer = MonotonicTimer().start()
    time.sleep(0.02)  # ~20ms sleep
    elapsed_ms = timer.stop()

    assert elapsed_ms >= 15.0  # Allow slight timing tolerance
    assert timer.elapsed_s() >= 0.015


def test_monotonic_timer_context_manager() -> None:
    """Verify MonotonicTimer works as a context manager."""
    with MonotonicTimer() as timer:
        time.sleep(0.01)
    assert timer.elapsed_ms() >= 8.0


def test_stage_latency_tracker_breakdown() -> None:
    """Verify StageLatencyTracker decomposes latency and supports Heading 4 diagnostics."""
    tracker = StageLatencyTracker(
        request_id="req_slow_042",
        scenario="sustained_overload",
        source="stand-in",
    )

    # 1. Simulate 50ms queue wait
    tracker.mark_queue_admitted(50.0)

    # 2. Simulate Stage 1 query analysis
    tracker.start_stage("stage1_query")
    time.sleep(0.01)
    tracker.stop_stage("stage1_query")

    # 3. Simulate Stage 2 retrieval
    tracker.start_stage("stage2_retrieval")
    time.sleep(0.01)
    tracker.stop_stage("stage2_retrieval")

    # 4. Simulate first token arrival (TTFT)
    ttft_recorded = tracker.record_first_token()
    assert ttft_recorded >= 20.0

    # 5. Simulate Stage 3 generate
    tracker.start_stage("stage3_generate")
    time.sleep(0.01)
    tracker.stop_stage("stage3_generate")

    # 6. Simulate Stage 4 verify
    tracker.start_stage("stage4_verify")
    time.sleep(0.01)
    tracker.stop_stage("stage4_verify")

    # 7. Finalize
    breakdown = tracker.finish_request()

    assert breakdown.waiting_ms == 50.0
    assert breakdown.working_ms > 30.0
    assert breakdown.retrying_ms == 0.0
    assert breakdown.pre_first_token_ms == round(ttft_recorded, 1)
    assert breakdown.post_first_token_ms is not None
    assert breakdown.post_first_token_ms >= 15.0

    # Export span record for Heading 4
    span = tracker.to_stage_span_record(slow_request_threshold_ms=40.0)
    assert span.request_id == "req_slow_042"
    assert span.slow_request_flag is True
    assert span.waiting_ms == 50.0
    assert span.working_ms == breakdown.working_ms


def test_atomic_queue_gauge_transitions() -> None:
    """Verify AtomicQueueGauge tracks depth, peak, and logs transitions."""
    gauge = AtomicQueueGauge()
    assert gauge.get_current_depth() == 0
    assert gauge.get_peak_depth() == 0

    # Enqueue 3 requests
    gauge.enqueue("req_1")
    gauge.enqueue("req_2")
    gauge.enqueue("req_3")
    assert gauge.get_current_depth() == 3
    assert gauge.get_peak_depth() == 3

    # Dequeue 1
    gauge.dequeue("req_1")
    assert gauge.get_current_depth() == 2
    assert gauge.get_peak_depth() == 3  # Peak remains 3!

    # Shed 1
    gauge.shed("req_rejected")
    assert gauge.get_current_depth() == 2

    # Timeout evict 1
    gauge.timeout_evict("req_2")
    assert gauge.get_current_depth() == 1

    # Check transition events
    transitions = gauge.get_transitions()
    assert len(transitions) == 6
    events = [t.event for t in transitions]
    assert events == ["enqueue", "enqueue", "enqueue", "dequeue", "shed", "timeout_evict"]

    # Export scenario metrics
    metrics = gauge.export_scenario_metrics(
        scenario="burst",
        source="stand-in",
        duration_s=20.0,
        total_requests=30,
        completed_requests=28,
        reconstructed_peak=3,
    )
    assert metrics.direct_peak_queue_length == 3
    assert metrics.reconstructed_peak_queue_length == 3
    assert metrics.queue_estimation_error == 0
    assert metrics.queue_reconstruction_valid is True


def test_secret_scrubbing_and_logger(tmp_path: Path) -> None:
    """Verify structured logger sanitizes secrets and writes clean JSONL traces."""
    secret_key = "gsk_testabcdef1234567890abcdef1234567890"

    # Test string scrubbing
    scrubbed = sanitize_message(f"Connecting with key {secret_key} now")
    assert secret_key not in scrubbed
    assert "[REDACTED_API_KEY]" in scrubbed

    # Test dictionary scrubbing
    dirty_dict = {
        "api_key": secret_key,
        "auth_header": f"Bearer {secret_key}",
        "regular_info": "normal_val",
        "nested": {"token_secret": secret_key},
    }
    clean_dict = sanitize_dict(dirty_dict)
    assert clean_dict["api_key"] == "[REDACTED]"
    assert secret_key not in str(clean_dict)

    # Test logger file output
    log_file = tmp_path / "test.log"
    spans_file = tmp_path / "stage_spans.jsonl"
    logger = StructuredLogger(
        component_name="test_component",
        log_file_path=log_file,
        spans_file_path=spans_file,
    )

    logger.info("request_received", request_id="req_001", key=secret_key)

    # Read back log file and verify scrubbing
    with open(log_file, "r", encoding="utf-8") as f:
        log_line = f.readline()
        log_data = json.loads(log_line)
        assert log_data["component"] == "test_component"
        assert log_data["event"] == "request_received"
        assert secret_key not in log_line
        assert log_data["metadata"]["key"] == "[REDACTED_API_KEY]"


def test_atomic_queue_gauge_reset() -> None:
    """Verify gauge reset clears depth and peak."""
    gauge = AtomicQueueGauge()
    gauge.enqueue("req_1")
    gauge.enqueue("req_2")
    assert gauge.get_peak_depth() == 2

    gauge.reset()
    assert gauge.get_current_depth() == 0
    assert gauge.get_peak_depth() == 0
    assert len(gauge.get_transitions()) == 0


def test_stage_latency_tracker_no_first_token() -> None:
    """Verify StageLatencyTracker handles requests shed before first token."""
    tracker = StageLatencyTracker(
        request_id="req_shed_001",
        scenario="burst",
        source="stand-in",
    )
    tracker.mark_queue_admitted(1.5)
    breakdown = tracker.finish_request()

    assert breakdown.waiting_ms == 1.5
    assert breakdown.pre_first_token_ms is None
    assert breakdown.post_first_token_ms is None

    span = tracker.to_stage_span_record()
    assert span.first_token_ms is None
    assert span.pre_ttft_delay_ms is None
    assert span.slow_request_flag is False

