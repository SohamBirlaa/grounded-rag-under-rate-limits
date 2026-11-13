"""Deterministic controllable stand-in inference engine for The Relay.

Provides local, reproducible LLM inference simulation without external API calls or credentials:
- Controllable TTFT with parametric prompt scaling, Gaussian noise, and slow TTFT mode
- Controllable inter-token generation cadence (ITL) with configurable chunk size and custom sleep_fn
- Controllable concurrency degradation tracking active in-flight requests against concurrency knee
- Multi-dimensional rate limit tracking (RPM, TPM, RPD, TPD) with realistic RFC-7807 429 payloads
- Deterministic synthetic answer generation grounded in input <passage> XML blocks
- Epistemic abstention on missing passages or unanswerable queries
- Controllable failure injection (transient 503, burst 429, daily exhaustion, timeouts, slow TTFT/TPOT)
- Configurable failure pattern counters (e.g. fail first N attempts then succeed)
- Fast-mode execution flag (inject_delay=False) for sub-second deterministic unit tests
- Strictly labeled as source="stand-in" in all telemetry and records
"""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import random
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from relay.config.settings import StandInConfig
from relay.contracts.errors import (
    DailyBudgetExhaustedError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
    ProviderQuotaSnapshot,
    ProviderStreamChunk,
)
from relay.rag.chunker import count_tokens

logger = logging.getLogger("relay.providers.standin")

PASSAGE_TAG_PATTERN = re.compile(
    r'<passage\s+id="(?P<id>[^"]+)"\s+paper_id="(?P<paper_id>[^"]+)"\s+'
    r'version="(?P<version>[^"]+)"\s+page="(?P<page>\d+)"\s+'
    r'section="(?P<section>[^"]+)"\s+content_type="(?P<type>[^"]+)">'
    r"(?P<content>.*?)"
    r"</passage>",
    re.DOTALL,
)


class StandInRateTracker:
    """Stateful rate-limit and quota tracker for Stand-In provider.

    Tracks requests per minute (RPM), tokens per minute (TPM),
    requests per day (RPD), and tokens per day (TPD).
    """

    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        daily_rpd: int = 14400,
        daily_tpd: int = 500000,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.daily_rpd = daily_rpd
        self.daily_tpd = daily_tpd
        self._clock_fn = clock_fn or time.monotonic
        self._request_timestamps: List[float] = []
        self._token_events: List[Tuple[float, int]] = []
        self._daily_requests_used: int = 0
        self._daily_tokens_used: int = 0

    def reset(self) -> None:
        """Reset all window and daily counters."""
        self._request_timestamps.clear()
        self._token_events.clear()
        self._daily_requests_used = 0
        self._daily_tokens_used = 0

    def _prune_windows(self, now: float) -> None:
        """Remove entries older than 60.0 seconds."""
        cutoff = now - 60.0
        self._request_timestamps = [t for t in self._request_timestamps if t >= cutoff]
        self._token_events = [(t, tok) for t, tok in self._token_events if t >= cutoff]

    def check_limits(self, estimated_tokens: int) -> None:
        """Check whether incoming request breaches daily or per-minute limits."""
        now = self._clock_fn()
        self._prune_windows(now)

        # 1. Daily Request Limit (RPD)
        if self._daily_requests_used >= self.daily_rpd:
            raise DailyBudgetExhaustedError(
                message=f"Stand-in simulated daily request quota exhaustion: Limit {self.daily_rpd} (RPD)",
                reset_time_utc="15:00:00Z",
            )

        # 2. Daily Token Limit (TPD)
        if self._daily_tokens_used + estimated_tokens > self.daily_tpd:
            raise DailyBudgetExhaustedError(
                message=f"Stand-in simulated daily token quota exhaustion: Limit {self.daily_tpd} (TPD)",
                reset_time_utc="15:00:00Z",
            )

        # 3. Requests Per Minute (RPM)
        if len(self._request_timestamps) >= self.rpm_limit:
            oldest = self._request_timestamps[0]
            retry_after = max(0.5, round(60.0 - (now - oldest), 2))
            raise RateLimitError(
                message=(
                    f"Rate limit reached on requests per minute (RPM): "
                    f"Limit {self.rpm_limit}, Used {len(self._request_timestamps)}. "
                    f"Please try again in {retry_after:.2f}s."
                ),
                retry_after_s=retry_after,
                limit_type="RPM",
                is_daily=False,
            )

        # 4. Tokens Per Minute (TPM)
        current_tpm = sum(tok for _, tok in self._token_events)
        if current_tpm + estimated_tokens > self.tpm_limit:
            oldest = self._token_events[0][0] if self._token_events else now
            retry_after = max(0.5, round(60.0 - (now - oldest), 2))
            raise RateLimitError(
                message=(
                    f"Rate limit reached on tokens per minute (TPM): "
                    f"Limit {self.tpm_limit}, Used {current_tpm}. "
                    f"Please try again in {retry_after:.2f}s."
                ),
                retry_after_s=retry_after,
                limit_type="TPM",
                is_daily=False,
            )

    def record_request_admitted(self, estimated_tokens: int) -> None:
        """Record admitted request arrival in sliding windows."""
        now = self._clock_fn()
        self._request_timestamps.append(now)
        self._token_events.append((now, estimated_tokens))
        self._daily_requests_used += 1
        self._daily_tokens_used += estimated_tokens

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        """Return snapshot of current remaining capacities."""
        now = self._clock_fn()
        self._prune_windows(now)
        current_tpm = sum(tok for _, tok in self._token_events)
        rpm_remaining = max(0, self.rpm_limit - len(self._request_timestamps))
        tpm_remaining = max(0, self.tpm_limit - current_tpm)
        rpd_remaining = max(0, self.daily_rpd - self._daily_requests_used)
        return ProviderQuotaSnapshot(
            rpd_limit=self.daily_rpd,
            rpd_remaining=rpd_remaining,
            rpd_reset_s=86400.0,
            tpm_limit=self.tpm_limit,
            tpm_remaining=tpm_remaining,
            tpm_reset_s=60.0,
            rpm_limit=self.rpm_limit,
            rpm_remaining=rpm_remaining,
        )


class StandInProviderAdapter:
    """Controllable local simulator for model inference."""

    def __init__(
        self,
        config: Optional[StandInConfig] = None,
        inject_delay: bool = True,
        failure_mode: Optional[str] = None,
        fail_attempts: int = 0,
        simulated_model: str = "llama-3.1-8b-instant-standin",
        sleep_fn: Optional[Callable[[float], Any]] = None,
        random_seed: Optional[int] = 42,
        enforce_rate_limits: bool = False,
    ) -> None:
        self.config = config or StandInConfig()
        self.inject_delay = inject_delay
        self.failure_mode = failure_mode
        self.fail_attempts = fail_attempts
        self.simulated_model = simulated_model
        self.enforce_rate_limits = enforce_rate_limits
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._attempt_counter = 0
        self._active_concurrency = 0

        self._rng = random.Random(random_seed) if random_seed is not None else None

        self.rate_tracker = StandInRateTracker(
            rpm_limit=self.config.rate_limit_rpm,
            tpm_limit=self.config.rate_limit_tpm,
            daily_rpd=self.config.daily_rpd,
            daily_tpd=self.config.daily_tpd,
        )

        self._quota_snapshot = self.rate_tracker.get_quota_snapshot()

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        """Return the current stand-in quota snapshot."""
        if self.enforce_rate_limits:
            return self.rate_tracker.get_quota_snapshot()
        return self._quota_snapshot

    def reset_quota_tracker(self) -> None:
        """Reset internal rate and quota tracker counters."""
        self.rate_tracker.reset()
        self._quota_snapshot = self.rate_tracker.get_quota_snapshot()

    async def check_health(self) -> bool:
        """Stand-in is always healthy unless configured with persistent server error."""
        return self.failure_mode != "server_error"

    def set_failure_mode(self, mode: Optional[str], fail_attempts: int = 0) -> None:
        """Configure dynamic failure injection for tests."""
        self.failure_mode = mode
        self.fail_attempts = fail_attempts
        self._attempt_counter = 0

    def get_active_concurrency(self) -> int:
        """Return the count of requests currently being served."""
        return self._active_concurrency

    def get_concurrency_degradation(self, active_count: Optional[int] = None) -> float:
        """Calculate latency degradation multiplier based on active concurrency."""
        c = self._active_concurrency if active_count is None else active_count
        knee = self.config.concurrency_knee
        if c <= knee:
            return 1.0
        return 1.0 + self.config.concurrency_slope * (c - knee)

    def calculate_simulated_ttft_ms(
        self, prompt_tokens: int, active_concurrency: Optional[int] = None
    ) -> float:
        """Calculate parametric TTFT based on prompt tokens, noise, and concurrency."""
        if self.failure_mode == "slow_ttft":
            return self.config.slow_ttft_ms

        base = self.config.base_ttft_ms
        prompt_delay = self.config.prompt_token_slope * prompt_tokens
        noise = 0.0
        if self._rng and self.config.noise_sigma_ms > 0 and self.inject_delay:
            noise = self._rng.gauss(0.0, self.config.noise_sigma_ms)

        raw_ttft = max(10.0, base + prompt_delay + noise)
        degradation = self.get_concurrency_degradation(active_concurrency)
        return raw_ttft * degradation

    def calculate_simulated_itl_ms(
        self, active_concurrency: Optional[int] = None
    ) -> float:
        """Calculate ITL cadence based on mode and concurrency degradation."""
        if self.failure_mode == "slow_tpot":
            base_itl = self.config.slow_tpot_ms
        else:
            base_itl = self.config.itl_ms
        degradation = self.get_concurrency_degradation(active_concurrency)
        return base_itl * degradation

    async def _sleep(self, seconds: float) -> None:
        """Awaitable sleep abstraction supporting virtual clocks and mock functions."""
        if seconds <= 0.0 or not self.inject_delay:
            return
        res = self._sleep_fn(seconds)
        if asyncio.iscoroutine(res):
            await res

    @asynccontextmanager
    async def _track_concurrency(self) -> AsyncIterator[int]:
        """Track in-flight concurrency count for load degradation modeling."""
        self._active_concurrency += 1
        try:
            yield self._active_concurrency
        finally:
            self._active_concurrency = max(0, self._active_concurrency - 1)

    def _should_fail(self) -> Tuple[bool, Optional[str]]:
        """Determine if current invocation should inject a configured failure."""
        if not self.failure_mode or self.failure_mode == "none":
            return False, None

        if self.fail_attempts > 0:
            if self._attempt_counter < self.fail_attempts:
                self._attempt_counter += 1
                return True, self.failure_mode
            return False, None

        return True, self.failure_mode

    def _extract_grounded_answer(self, user_prompt: str) -> Tuple[str, bool]:
        """Synthesize a deterministic grounded response from input <passage> XML.

        Returns (answer_text, is_refusal).
        """
        # Check if prompt contains unanswerable indications or missing passages
        if "<!-- No relevant passages retrieved -->" in user_prompt:
            return (
                "I cannot answer this question based on the provided corpus. "
                "The relevant information is not present in the retrieved research documents.",
                True,
            )

        passages = list(PASSAGE_TAG_PATTERN.finditer(user_prompt))
        if not passages:
            # Fallback if no passages matched
            return (
                "I cannot answer this question based on the provided corpus. "
                "The relevant information is not present in the retrieved research documents.",
                True,
            )

        # Build grounded answer from the first 1-2 retrieved passages
        p0 = passages[0]
        cid0 = p0.group("id")
        paper0 = p0.group("paper_id")
        page0 = int(p0.group("page"))
        sec0 = p0.group("section")
        content0 = p0.group("content").strip()

        # Extract representative sentence from passage content
        sentences = [s.strip() for s in content0.split(".") if len(s.strip()) > 15]
        evidence_snippet = sentences[0] if sentences else content0[:80]

        answer = (
            f"Based on the research corpus, {evidence_snippet}. "
            f"[arXiv:{paper0} p.{page0} §{sec0} ({cid0})]"
        )

        return answer, False

    async def stream_completion(
        self, request: ProviderCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        """Stream simulated completion chunks matching Groq provider interface."""
        # 1. Check failure mode injection
        should_fail, mode = self._should_fail()
        if should_fail:
            if mode == "auth_error":
                raise ProviderAuthenticationError(
                    "Stand-in simulated auth rejection: HTTP 401 Unauthorized"
                )
            elif mode == "rate_limit_burst":
                raise RateLimitError(
                    message="Stand-in simulated burst rate limit: Limit 30, Used 30 (RPM)",
                    retry_after_s=1.5,
                    limit_type="RPM",
                    is_daily=False,
                )
            elif mode == "rate_limit_tpm":
                raise RateLimitError(
                    message="Stand-in simulated TPM rate limit: Limit 6000, Used 6000 (TPM)",
                    retry_after_s=2.0,
                    limit_type="TPM",
                    is_daily=False,
                )
            elif mode == "daily_exhausted":
                raise DailyBudgetExhaustedError(
                    message="Stand-in simulated daily quota exhaustion: Limit 14400 (RPD)",
                    reset_time_utc="15:00:00Z",
                )
            elif mode == "timeout":
                if self.inject_delay:
                    await self._sleep(request.timeout_s + 0.5)
                raise ProviderTimeoutError(
                    f"Stand-in simulated operation timeout exceeding {request.timeout_s}s"
                )
            elif mode == "server_error":
                raise ProviderUnavailableError(
                    "Stand-in simulated provider failure: HTTP 503 Service Unavailable",
                    status_code=503,
                )
            elif mode == "transient_network":
                raise ProviderUnavailableError(
                    "Stand-in simulated transient network connection reset",
                    status_code=None,
                )
            elif mode in ("slow_ttft", "slow_tpot"):
                # Non-throwing modes; handled during latency calculation
                pass
            else:
                raise ProviderUnavailableError(f"Stand-in unknown simulated failure: {mode}")

        # Extract prompt text
        user_content = ""
        for m in request.messages:
            if m.get("role") == "user":
                user_content += m.get("content", "") + " "

        answer_text, _ = self._extract_grounded_answer(user_content)
        prompt_tokens = count_tokens(user_content)
        completion_tokens = count_tokens(answer_text)
        estimated_total_tokens = prompt_tokens + completion_tokens

        # Check rate tracker if enabled
        if self.enforce_rate_limits:
            self.rate_tracker.check_limits(estimated_total_tokens)
            self.rate_tracker.record_request_admitted(estimated_total_tokens)
            self._quota_snapshot = self.rate_tracker.get_quota_snapshot()

        async with self._track_concurrency() as active_c:
            # Simulate TTFT delay
            simulated_ttft_ms = self.calculate_simulated_ttft_ms(prompt_tokens, active_c)
            if self.inject_delay and simulated_ttft_ms > 0:
                await self._sleep(simulated_ttft_ms / 1000.0)

            # Split answer into small token words (chunk_size words per SSE chunk)
            words = answer_text.split()
            chunk_size = max(1, self.config.chunk_size)
            token_index = 0
            is_first = True

            effective_itl_ms = self.calculate_simulated_itl_ms(active_c)

            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])
                if i + chunk_size < len(words):
                    chunk_text += " "

                token_index += 1
                yield ProviderStreamChunk(
                    text_delta=chunk_text,
                    is_first_token=is_first,
                    token_index=token_index,
                    arrival_monotonic_s=time.perf_counter(),
                    prompt_tokens=prompt_tokens if (i + chunk_size >= len(words)) else None,
                    completion_tokens=completion_tokens if (i + chunk_size >= len(words)) else None,
                    finish_reason="stop" if (i + chunk_size >= len(words)) else None,
                )

                if is_first:
                    is_first = False

                # Simulate ITL delay between chunks
                if self.inject_delay and effective_itl_ms > 0:
                    await self._sleep((effective_itl_ms * chunk_size) / 1000.0)
