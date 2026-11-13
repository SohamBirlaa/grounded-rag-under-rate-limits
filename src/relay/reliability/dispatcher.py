"""Event-driven admission dispatcher coordinating BoundedRequestQueue and ConcurrencyGate.

Implements ADR-04, ADR-08, and Inference Design Section 4.3:
- Maintains strict separation of admission vs. execution (caller coroutine executes Stages 1-4).
- Regulates in-flight admissions against ConcurrencyGate.max_llm_concurrency.
- Signals waiting requests strictly in FIFO order by calling BoundedRequestQueue.pop_next().
- Non-blocking event-driven dispatch pumping on arrival, completion, and concurrency changes.
- Supervisory background loop with clean startup/shutdown lifecycle and zero task leaks.
"""

import asyncio
import logging
from typing import Optional

from relay.reliability.concurrency import ConcurrencyGate
from relay.reliability.queue import BoundedRequestQueue

logger = logging.getLogger("relay.reliability.dispatcher")


class QueueAdmissionDispatcher:
    """Event-driven admission dispatcher coordinating BoundedRequestQueue and ConcurrencyGate."""

    def __init__(
        self,
        queue: BoundedRequestQueue,
        concurrency_gate: ConcurrencyGate,
    ) -> None:
        self.queue = queue
        self.concurrency_gate = concurrency_gate
        self._active_admissions = 0
        self._lock = asyncio.Lock()
        self._wake_event = asyncio.Event()
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def active_admissions(self) -> int:
        """Current number of admitted requests actively executing in the pipeline."""
        return self._active_admissions

    @property
    def is_running(self) -> bool:
        """Whether the supervisory background loop is actively running."""
        return (
            self._running
            and self._dispatcher_task is not None
            and not self._dispatcher_task.done()
        )

    async def start(self) -> None:
        """Start the background admission dispatcher supervisory task."""
        async with self._lock:
            if self._running and self._dispatcher_task and not self._dispatcher_task.done():
                return
            self._running = True
            self._wake_event.clear()
            self._dispatcher_task = asyncio.create_task(
                self._dispatch_loop(), name="relay_queue_admission_dispatcher"
            )
            logger.info("Started QueueAdmissionDispatcher supervisory loop")

    async def stop(self) -> None:
        """Stop the background admission dispatcher loop cleanly with zero task leaks."""
        async with self._lock:
            self._running = False
            task = self._dispatcher_task
            self._dispatcher_task = None
            self._wake_event.set()

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning("Error during QueueAdmissionDispatcher shutdown: %s", e)
            logger.info("Stopped QueueAdmissionDispatcher supervisory loop")

    def notify(self) -> None:
        """Signal the dispatcher that queue depth or concurrency capacity changed."""
        self._wake_event.set()

    async def pump_admissions(self) -> int:
        """Admit waiting requests from the FIFO queue as long as in-flight capacity is available.

        Returns:
            Number of requests admitted in this pump iteration.
        """
        admitted = 0
        async with self._lock:
            max_concurrency = self.concurrency_gate.max_llm_concurrency
            while self._active_admissions < max_concurrency and self.queue.current_depth > 0:
                item = await self.queue.pop_next()
                if item is None:
                    break
                self._active_admissions += 1
                admitted += 1
        return admitted

    async def release_slot(self) -> None:
        """Release an admission slot when an admitted request finishes execution.

        Immediately pumps the queue to admit the next waiting request in FIFO order.
        """
        async with self._lock:
            self._active_admissions = max(0, self._active_admissions - 1)
        self.notify()
        await self.pump_admissions()

    async def _dispatch_loop(self) -> None:
        """Supervisory event-driven loop that pumps admissions whenever signaled."""
        while self._running:
            try:
                await self._wake_event.wait()
                self._wake_event.clear()
                if not self._running:
                    break
                await self.pump_admissions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Unexpected error in QueueAdmissionDispatcher: %s", e)
                await asyncio.sleep(0.05)

