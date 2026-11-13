"""Bounded FIFO request queue with atomic queue depth gauge and proactive deadline shedding.

Implements ADR-08 and Inference Design Section 4.3:
- Bounded FIFO queue: max_size = 24
- Atomic queue depth tracking via AtomicQueueGauge
- Proactive wait-time deadline shedding based on service rate R_eff
- Queue timeout eviction after queue_timeout_ms (4500ms)
- Records true physical queue transitions for validation against CSV sweepline reconstruction
"""

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from relay.config.settings import ReliabilityConfig
from relay.contracts.errors import (
    DeadlineShedError,
    QueueFullError,
    QueueTimeoutError,
    RequestDeadlineExceededError,
)
from relay.observability.metrics import AtomicQueueGauge
from relay.reliability.timeouts import DeadlineTracker

logger = logging.getLogger("relay.reliability.queue")


@dataclass
class QueuedRequest:
    """Internal record for a request waiting in the bounded FIFO queue."""

    request_id: str
    arrival_monotonic_s: float
    deadline_tracker: DeadlineTracker
    admitted_event: asyncio.Event = field(default_factory=asyncio.Event)
    evicted: bool = False
    eviction_reason: Optional[str] = None


class BoundedRequestQueue:
    """Bounded FIFO admission queue with proactive shedding and atomic gauge instrumentation."""

    def __init__(
        self,
        config: Optional[ReliabilityConfig] = None,
        gauge: Optional[AtomicQueueGauge] = None,
        clock_fn: Optional[Callable[[], float]] = None,
        default_service_rate_req_per_s: float = 2.0,
        min_exec_time_s: float = 0.3,
    ) -> None:
        self.config = config or ReliabilityConfig()
        self.max_queue_size = self.config.max_queue_size
        self.queue_timeout_s = self.config.queue_timeout_ms / 1000.0
        self.gauge = gauge or AtomicQueueGauge()
        self._clock_fn = clock_fn or time.perf_counter
        self._min_exec_time_s = min_exec_time_s

        self._queue: List[QueuedRequest] = []
        self._lock = asyncio.Lock()

        # Service rate tracking for proactive deadline estimation (R_eff)
        self._completed_service_times: List[float] = []
        self._default_service_rate = default_service_rate_req_per_s

    @property
    def current_depth(self) -> int:
        """Current physical depth of the waiting queue."""
        return self.gauge.depth

    @property
    def peak_depth(self) -> int:
        """Maximum queue depth observed since initialization."""
        return self.gauge.peak_depth

    @property
    def effective_service_rate(self) -> float:
        """Estimated completed service rate in requests per second."""
        if len(self._completed_service_times) >= 3:
            avg_service_time = sum(self._completed_service_times[-10:]) / len(
                self._completed_service_times[-10:]
            )
            if avg_service_time > 0.01:
                return 1.0 / avg_service_time
        return self._default_service_rate

    def record_service_completion(self, service_duration_s: float) -> None:
        """Record completed request duration to update estimated service rate."""
        if service_duration_s > 0.0:
            self._completed_service_times.append(service_duration_s)
            if len(self._completed_service_times) > 50:
                self._completed_service_times.pop(0)

    async def admit_or_shed(
        self,
        request_id: str,
        deadline_tracker: DeadlineTracker,
    ) -> QueuedRequest:
        """Evaluate admission criteria: shed if queue full or if wait exceeds deadline.

        Raises:
            QueueFullError: If current queue depth >= max_queue_size.
            DeadlineShedError: If estimated queue wait + min_exec > remaining deadline.
        """
        async with self._lock:
            # 1. Capacity Ceiling Check
            if self.gauge.depth >= self.max_queue_size:
                self.gauge.shed(request_id)
                logger.warning(
                    "Request %s shed: queue full (depth %d >= max %d)",
                    request_id,
                    self.gauge.depth,
                    self.max_queue_size,
                )
                raise QueueFullError(
                    f"Request queue capacity reached ceiling ({self.max_queue_size})"
                )

            # 2. Proactive Deadline Shedding Check (§4.3.2)
            # Estimated wait: W_hat = (Q + 1) / R_eff
            r_eff = self.effective_service_rate
            estimated_wait_s = (self.gauge.depth + 1) / r_eff
            remaining_deadline_s = deadline_tracker.remaining_s

            if estimated_wait_s + self._min_exec_time_s > remaining_deadline_s:
                self.gauge.shed(request_id)
                logger.warning(
                    "Request %s shed proactively: estimated wait %.2fs + min_exec %.2fs > deadline %.2fs",
                    request_id,
                    estimated_wait_s,
                    self._min_exec_time_s,
                    remaining_deadline_s,
                )
                raise DeadlineShedError(
                    f"Shed proactively: estimated wait {estimated_wait_s:.2f}s + min exec {self._min_exec_time_s:.2f}s "
                    f"exceeds remaining deadline {remaining_deadline_s:.2f}s"
                )

            # 3. Admitted to FIFO Queue
            item = QueuedRequest(
                request_id=request_id,
                arrival_monotonic_s=self._clock_fn(),
                deadline_tracker=deadline_tracker,
            )
            self._queue.append(item)
            self.gauge.enqueue(request_id)
            return item

    async def wait_for_admission(self, queued_req: QueuedRequest) -> float:
        """Wait in queue until admitted by worker or evicted by timeout.

        Returns:
            Queue wait duration in milliseconds.

        Raises:
            QueueTimeoutError: If waiting time exceeds queue_timeout_ms.
            RequestDeadlineExceededError: If client deadline expires while waiting.
        """
        now = self._clock_fn()
        elapsed_wait = now - queued_req.arrival_monotonic_s
        max_wait_allowed = min(
            self.queue_timeout_s - elapsed_wait,
            queued_req.deadline_tracker.remaining_s,
        )

        if max_wait_allowed <= 0.0:
            self._evict_item(queued_req, "timeout")
            if queued_req.deadline_tracker.is_expired:
                raise RequestDeadlineExceededError(
                    f"Deadline of {queued_req.deadline_tracker.deadline_ms}ms expired in queue",
                    deadline_ms=queued_req.deadline_tracker.deadline_ms,
                    elapsed_ms=queued_req.deadline_tracker.timer.elapsed_ms(),
                )
            raise QueueTimeoutError(
                f"Request {queued_req.request_id} timed out waiting in queue after {elapsed_wait * 1000.0:.1f}ms"
            )

        try:
            await asyncio.wait_for(queued_req.admitted_event.wait(), timeout=max_wait_allowed)
        except (asyncio.TimeoutError, TimeoutError):
            if queued_req.admitted_event.is_set():
                waiting_ms = (self._clock_fn() - queued_req.arrival_monotonic_s) * 1000.0
                return max(0.0, waiting_ms)
            self._evict_item(queued_req, "timeout")
            if queued_req.deadline_tracker.is_expired:
                raise RequestDeadlineExceededError(
                    f"Deadline of {queued_req.deadline_tracker.deadline_ms}ms expired in queue",
                    deadline_ms=queued_req.deadline_tracker.deadline_ms,
                    elapsed_ms=queued_req.deadline_tracker.timer.elapsed_ms(),
                )
            total_waited = (self._clock_fn() - queued_req.arrival_monotonic_s) * 1000.0
            raise QueueTimeoutError(
                f"Request {queued_req.request_id} timed out waiting in queue after {total_waited:.1f}ms"
            )
        except asyncio.CancelledError:
            if not queued_req.admitted_event.is_set():
                self._evict_item(queued_req, "cancelled")
            raise

        # Successfully popped from queue
        waiting_ms = (self._clock_fn() - queued_req.arrival_monotonic_s) * 1000.0
        return max(0.0, waiting_ms)

    async def pop_next(self) -> Optional[QueuedRequest]:
        """Pop the next waiting request from the head of the FIFO queue."""
        async with self._lock:
            while self._queue:
                item = self._queue.pop(0)
                if not item.evicted:
                    self.gauge.dequeue(item.request_id)
                    item.admitted_event.set()
                    return item
            return None

    def _evict_item(self, queued_req: QueuedRequest, reason: str) -> None:
        """Mark an item as evicted and update gauge counter."""
        if queued_req.evicted:
            return
        queued_req.evicted = True
        queued_req.eviction_reason = reason
        self.gauge.timeout_evict(queued_req.request_id)
        if queued_req in self._queue:
            self._queue.remove(queued_req)
