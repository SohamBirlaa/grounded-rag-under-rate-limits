"""Concurrency gates and worker semaphore management for The Relay.

Implements ADR-04 and Inference Design Section 4.3.1:
- Model call concurrency gate: asyncio.Semaphore(max_in_flight_llm)
- Tool retrieval concurrency gate: asyncio.Semaphore(max_in_flight_tools)
- Dynamic limit adjustment (e.g. lowering LLM concurrency to 4 under rpm_halved)
- Safe slot release on cancellation or exception in finally block
"""

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator, Optional

from relay.config.settings import ReliabilityConfig

logger = logging.getLogger("relay.reliability.concurrency")


class ConcurrencyGate:
    """Manages concurrent in-flight model calls and tool executions."""

    def __init__(self, config: Optional[ReliabilityConfig] = None) -> None:
        self.config = config or ReliabilityConfig()
        self._max_llm = self.config.max_in_flight_llm
        self._max_tools = self.config.max_in_flight_tools

        self._llm_semaphore = asyncio.Semaphore(self._max_llm)
        self._tool_semaphore = asyncio.Semaphore(self._max_tools)

        self._active_llm = 0
        self._active_tools = 0
        self._lock = asyncio.Lock()

    @property
    def max_llm_concurrency(self) -> int:
        """Current configured maximum in-flight LLM calls."""
        return self._max_llm

    @property
    def max_tool_concurrency(self) -> int:
        """Current configured maximum in-flight retrieval tool calls."""
        return self._max_tools

    @property
    def active_llm_count(self) -> int:
        """Current active in-flight LLM calls."""
        return self._active_llm

    @property
    def active_tool_count(self) -> int:
        """Current active in-flight retrieval tool calls."""
        return self._active_tools

    def set_llm_limit(self, new_limit: int) -> None:
        """Dynamically adjust LLM concurrency limit (e.g. for rpm_halved scenario)."""
        if new_limit < 1:
            raise ValueError(f"Concurrency limit must be >= 1, got {new_limit}")
        self._max_llm = new_limit
        self._llm_semaphore = asyncio.Semaphore(new_limit)
        logger.info("Updated max_in_flight_llm to %d", new_limit)

    def set_tool_limit(self, new_limit: int) -> None:
        """Dynamically adjust tool concurrency limit."""
        if new_limit < 1:
            raise ValueError(f"Tool concurrency limit must be >= 1, got {new_limit}")
        self._max_tools = new_limit
        self._tool_semaphore = asyncio.Semaphore(new_limit)
        logger.info("Updated max_in_flight_tools to %d", new_limit)

    @asynccontextmanager
    async def acquire_llm(self, timeout_s: Optional[float] = None) -> AsyncIterator[int]:
        """Acquire LLM call concurrency slot."""
        if timeout_s is not None and timeout_s > 0:
            await asyncio.wait_for(self._llm_semaphore.acquire(), timeout=timeout_s)
        else:
            await self._llm_semaphore.acquire()

        async with self._lock:
            self._active_llm += 1
            current = self._active_llm
        try:
            yield current
        finally:
            async with self._lock:
                self._active_llm = max(0, self._active_llm - 1)
            self._llm_semaphore.release()

    @asynccontextmanager
    async def acquire_tool(self, timeout_s: Optional[float] = None) -> AsyncIterator[int]:
        """Acquire retrieval tool concurrency slot."""
        if timeout_s is not None and timeout_s > 0:
            await asyncio.wait_for(self._tool_semaphore.acquire(), timeout=timeout_s)
        else:
            await self._tool_semaphore.acquire()

        async with self._lock:
            self._active_tools += 1
            current = self._active_tools
        try:
            yield current
        finally:
            async with self._lock:
                self._active_tools = max(0, self._active_tools - 1)
            self._tool_semaphore.release()
