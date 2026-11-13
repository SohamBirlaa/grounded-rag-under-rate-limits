"""High-resolution monotonic timers, stage trackers, and atomic queue gauges.

Implements client-side telemetry capture without claiming access to Groq's internal
hardware counters or scheduler states.
"""

import threading
import time
from typing import Any, Dict, List, Optional
from relay.contracts.schemas import (
    LatencyBreakdown,
    QueueTransitionEvent,
    ScenarioMetricsRecord,
    StageSpanRecord,
)


class MonotonicTimer:
    """Nanosecond-resolution monotonic timer using time.perf_counter_ns()."""

    def __init__(self) -> None:
        self._start_ns: Optional[int] = None
        self._stop_ns: Optional[int] = None

    def start(self) -> "MonotonicTimer":
        self._start_ns = time.perf_counter_ns()
        self._stop_ns = None
        return self

    def stop(self) -> float:
        if self._start_ns is None:
            raise RuntimeError("Timer was never started")
        self._stop_ns = time.perf_counter_ns()
        return self.elapsed_ms()

    def elapsed_ns(self) -> int:
        if self._start_ns is None:
            return 0
        end = self._stop_ns if self._stop_ns is not None else time.perf_counter_ns()
        return end - self._start_ns

    def elapsed_ms(self) -> float:
        return self.elapsed_ns() / 1_000_000.0

    def elapsed_s(self) -> float:
        return self.elapsed_ns() / 1_000_000_000.0

    def __enter__(self) -> "MonotonicTimer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class StageLatencyTracker:
    """Tracks per-stage execution durations and decomposes request latency.

    Specifically engineered to provide the quantitative proofs required by Heading 4:
    total_ms = waiting_ms + working_ms + retrying_ms
    pre_ttft_delay vs post_ttft_delay attribution
    """

    def __init__(self, request_id: str, scenario: str, source: str) -> None:
        self.request_id = request_id
        self.scenario = scenario
        self.source = source
        self.start_timer = MonotonicTimer().start()
        self.start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"

        self._waiting_ms: float = 0.0
        self._working_ms: float = 0.0
        self._retrying_ms: float = 0.0

        self._stage_timers: Dict[str, MonotonicTimer] = {}
        self._stage_durations: Dict[str, float] = {
            "stage1_query": 0.0,
            "stage2_retrieval": 0.0,
            "stage3_generate": 0.0,
            "stage4_verify": 0.0,
        }

        self._first_token_ms: Optional[float] = None
        self._total_ms: Optional[float] = None

    def mark_queue_admitted(self, wait_duration_ms: float) -> None:
        """Record queue wait duration."""
        self._waiting_ms = max(0.0, wait_duration_ms)

    def start_stage(self, stage_name: str) -> None:
        """Start a stage timer."""
        self._stage_timers[stage_name] = MonotonicTimer().start()

    def stop_stage(self, stage_name: str) -> float:
        """Stop a stage timer and record duration."""
        timer = self._stage_timers.get(stage_name)
        if timer is not None:
            dur = timer.stop()
            self._stage_durations[stage_name] = dur
            self._working_ms += dur
            return dur
        return 0.0

    def record_retry(self, retry_duration_ms: float) -> None:
        """Accumulate failed attempt duration + retry backoff delay."""
        self._retrying_ms += max(0.0, retry_duration_ms)

    def record_first_token(self) -> float:
        """Record the exact time when the first token chunk arrived."""
        if self._first_token_ms is None:
            self._first_token_ms = self.start_timer.elapsed_ms()
        return self._first_token_ms

    def finish_request(self) -> LatencyBreakdown:
        """Finalize request timing and return LatencyBreakdown."""
        self._total_ms = self.start_timer.stop()

        pre_ttft = self._first_token_ms
        post_ttft = (self._total_ms - self._first_token_ms) if self._first_token_ms is not None else None

        return LatencyBreakdown(
            waiting_ms=round(self._waiting_ms, 1),
            working_ms=round(self._working_ms, 1),
            retrying_ms=round(self._retrying_ms, 1),
            stage1_query_ms=round(self._stage_durations.get("stage1_query", 0.0), 1),
            stage2_retrieval_ms=round(self._stage_durations.get("stage2_retrieval", 0.0), 1),
            stage3_generate_ms=round(self._stage_durations.get("stage3_generate", 0.0), 1),
            stage4_verify_ms=round(self._stage_durations.get("stage4_verify", 0.0), 1),
            pre_first_token_ms=round(pre_ttft, 1) if pre_ttft is not None else None,
            post_first_token_ms=round(post_ttft, 1) if post_ttft is not None else None,
        )

    def to_stage_span_record(self, slow_request_threshold_ms: float = 3000.0) -> StageSpanRecord:
        """Export full stage breakdown record for stage_spans.jsonl."""
        breakdown = self.finish_request() if self._total_ms is None else LatencyBreakdown(
            waiting_ms=round(self._waiting_ms, 1),
            working_ms=round(self._working_ms, 1),
            retrying_ms=round(self._retrying_ms, 1),
            stage1_query_ms=round(self._stage_durations.get("stage1_query", 0.0), 1),
            stage2_retrieval_ms=round(self._stage_durations.get("stage2_retrieval", 0.0), 1),
            stage3_generate_ms=round(self._stage_durations.get("stage3_generate", 0.0), 1),
            stage4_verify_ms=round(self._stage_durations.get("stage4_verify", 0.0), 1),
            pre_first_token_ms=round(self._first_token_ms, 1) if self._first_token_ms is not None else None,
            post_first_token_ms=round(self._total_ms - self._first_token_ms, 1) if self._first_token_ms is not None and self._total_ms is not None else None,
        )
        total = self._total_ms or self.start_timer.elapsed_ms()

        return StageSpanRecord(
            request_id=self.request_id,
            scenario=self.scenario,
            source=self.source,
            start_time_iso=self.start_time_iso,
            waiting_ms=breakdown.waiting_ms,
            working_ms=breakdown.working_ms,
            retrying_ms=breakdown.retrying_ms,
            stage1_query_ms=breakdown.stage1_query_ms or 0.0,
            stage2_retrieval_ms=breakdown.stage2_retrieval_ms or 0.0,
            stage3_generate_ms=breakdown.stage3_generate_ms or 0.0,
            stage4_verify_ms=breakdown.stage4_verify_ms or 0.0,
            first_token_ms=self._first_token_ms,
            total_ms=round(total, 1),
            pre_ttft_delay_ms=breakdown.pre_first_token_ms,
            post_ttft_delay_ms=breakdown.post_first_token_ms,
            slow_request_flag=(total >= slow_request_threshold_ms),
        )


class AtomicQueueGauge:
    """Thread-safe and async-compatible atomic queue depth gauge.

    Tracks true instantaneous physical queue depth Q_true(t), records peak queue depth,
    and logs transition events for validation against CSV sweepline reconstruction.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._peak_depth = 0
        self._events: List[QueueTransitionEvent] = []

    def enqueue(self, request_id: str) -> int:
        """Record arrival enqueued in FIFO queue."""
        with self._lock:
            self._depth += 1
            if self._depth > self._peak_depth:
                self._peak_depth = self._depth
            current = self._depth
            self._events.append(
                QueueTransitionEvent(
                    timestamp_ms=time.time() * 1000.0,
                    event="enqueue",
                    queue_depth=current,
                    request_id=request_id,
                )
            )
            return current

    def dequeue(self, request_id: str) -> int:
        """Record request popped from FIFO queue for active processing."""
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            current = self._depth
            self._events.append(
                QueueTransitionEvent(
                    timestamp_ms=time.time() * 1000.0,
                    event="dequeue",
                    queue_depth=current,
                    request_id=request_id,
                )
            )
            return current

    def shed(self, request_id: str) -> int:
        """Record request shed immediately due to queue capacity or deadline."""
        with self._lock:
            current = self._depth
            self._events.append(
                QueueTransitionEvent(
                    timestamp_ms=time.time() * 1000.0,
                    event="shed",
                    queue_depth=current,
                    request_id=request_id,
                )
            )
            return current

    def timeout_evict(self, request_id: str) -> int:
        """Record request evicted after exceeding queue_timeout_ms."""
        with self._lock:
            if self._depth > 0:
                self._depth -= 1
            current = self._depth
            self._events.append(
                QueueTransitionEvent(
                    timestamp_ms=time.time() * 1000.0,
                    event="timeout_evict",
                    queue_depth=current,
                    request_id=request_id,
                )
            )
            return current

    @property
    def depth(self) -> int:
        """Current instantaneous queue depth."""
        return self.get_current_depth()

    @property
    def peak_depth(self) -> int:
        """Maximum observed peak queue depth."""
        return self.get_peak_depth()

    @property
    def total_shed(self) -> int:
        """Total requests shed due to capacity or deadline."""
        with self._lock:
            return sum(1 for e in self._events if e.event == "shed")

    @property
    def total_evicted(self) -> int:
        """Total requests evicted due to queue timeout."""
        with self._lock:
            return sum(1 for e in self._events if e.event == "timeout_evict")

    def get_current_depth(self) -> int:
        """Return current instantaneous queue depth."""
        with self._lock:
            return self._depth

    def get_peak_depth(self) -> int:
        """Return maximum observed peak queue depth."""
        with self._lock:
            return self._peak_depth

    def get_transitions(self) -> List[QueueTransitionEvent]:
        """Return copy of transition events."""
        with self._lock:
            return list(self._events)

    def reset(self) -> None:
        """Reset gauge counters between benchmark runs."""
        with self._lock:
            self._depth = 0
            self._peak_depth = 0
            self._events.clear()

    def export_scenario_metrics(
        self,
        scenario: str,
        source: str,
        duration_s: float,
        total_requests: int,
        completed_requests: int,
        reconstructed_peak: Optional[int] = None,
    ) -> ScenarioMetricsRecord:
        """Generate ScenarioMetricsRecord for data/traces/scenario_metrics.json."""
        with self._lock:
            peak_true = self._peak_depth
            count = len(self._events)

        error: Optional[int] = None
        valid = True
        if reconstructed_peak is not None:
            error = abs(peak_true - reconstructed_peak)
            valid = error <= 2

        return ScenarioMetricsRecord(
            scenario=scenario,
            source=source,
            duration_s=round(duration_s, 1),
            total_requests=total_requests,
            completed_requests=completed_requests,
            direct_peak_queue_length=peak_true,
            reconstructed_peak_queue_length=reconstructed_peak,
            queue_estimation_error=error,
            queue_reconstruction_valid=valid,
            transitions_count=count,
        )
