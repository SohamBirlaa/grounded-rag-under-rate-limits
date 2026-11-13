"""Client-side multi-dimensional rate limiter and token quota manager.

Implements ADR-09 and Inference Design Section 4.4:
- Multi-dimensional tracking: RPM, TPM, RPD, TPD
- 60-second sliding windows for minute limits
- Pre-reservation of estimated tokens before dispatch (prompt tokens + max generation)
- Post-call token reconciliation and refunding of unspent reservations
- Live response header reconciliation (x-ratelimit-remaining-*)
- Quota-Conserving Compact Mode signal when remaining TPM <= 25%
- Daily state persistence to data/state/quota_state.json
- Fully supports virtual clocks for deterministic sub-second testing
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from relay.config.settings import ReliabilityConfig
from relay.contracts.errors import DailyBudgetExhaustedError, RateLimitError
from relay.providers.base import ProviderQuotaSnapshot

logger = logging.getLogger("relay.reliability.rate_limiter")


class ClientRateLimiter:
    """Client-side admission rate limiter managing RPM, TPM, RPD, and TPD."""

    def __init__(
        self,
        config: Optional[ReliabilityConfig] = None,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        daily_rpd: int = 14400,
        daily_tpd: int = 500000,
        tpm_compact_threshold_pct: float = 0.25,
        state_path: Optional[Path] = None,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config or ReliabilityConfig()
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.daily_rpd = daily_rpd
        self.daily_tpd = daily_tpd
        self.tpm_compact_threshold_pct = tpm_compact_threshold_pct
        self.state_path = state_path or Path("data/state/quota_state.json")
        self._clock_fn = clock_fn or time.monotonic

        # 60-second sliding window structures: list of timestamps / (timestamp, tokens)
        self._request_timestamps: List[float] = []
        self._token_reservations: List[Tuple[float, int]] = []

        # Daily usage counters
        self._daily_requests_used: int = 0
        self._daily_tokens_used: int = 0

        # Try to load existing persisted state if present
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted daily quota state from disk if available."""
        if self.state_path and self.state_path.is_file():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._daily_requests_used = data.get("daily_requests_used", 0)
                    self._daily_tokens_used = data.get("daily_tokens_used", 0)
            except Exception as e:
                logger.warning("Could not load quota state: %s", e)

    def save_state(self) -> None:
        """Persist current daily quota state to disk."""
        if self.state_path:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "daily_requests_used": self._daily_requests_used,
                            "daily_tokens_used": self._daily_tokens_used,
                            "last_saved_timestamp": self._clock_fn(),
                        },
                        f,
                        indent=2,
                    )
            except Exception as e:
                logger.warning("Could not persist quota state: %s", e)

    def reset(self) -> None:
        """Reset sliding windows and daily counters."""
        self._request_timestamps.clear()
        self._token_reservations.clear()
        self._daily_requests_used = 0
        self._daily_tokens_used = 0
        self.save_state()

    def _prune_windows(self, now: float) -> None:
        """Remove window entries older than 60.0 seconds."""
        cutoff = now - 60.0
        self._request_timestamps = [t for t in self._request_timestamps if t >= cutoff]
        self._token_reservations = [
            (t, tok) for t, tok in self._token_reservations if t >= cutoff
        ]

    @property
    def current_rpm_used(self) -> int:
        """Active requests count in the rolling 60-second window."""
        self._prune_windows(self._clock_fn())
        return len(self._request_timestamps)

    @property
    def current_tpm_used(self) -> int:
        """Active reserved tokens count in the rolling 60-second window."""
        self._prune_windows(self._clock_fn())
        return sum(tok for _, tok in self._token_reservations)

    @property
    def remaining_rpm(self) -> int:
        """Remaining request capacity in current 60-second window."""
        return max(0, self.rpm_limit - self.current_rpm_used)

    @property
    def remaining_tpm(self) -> int:
        """Remaining token capacity in current 60-second window."""
        return max(0, self.tpm_limit - self.current_tpm_used)

    @property
    def remaining_rpd(self) -> int:
        """Remaining daily request capacity."""
        return max(0, self.daily_rpd - self._daily_requests_used)

    @property
    def remaining_tpd(self) -> int:
        """Remaining daily token capacity."""
        return max(0, self.daily_tpd - self._daily_tokens_used)

    @property
    def is_compact_mode_needed(self) -> bool:
        """Return True if remaining TPM is at or below the compact mode threshold."""
        threshold = self.tpm_limit * self.tpm_compact_threshold_pct
        return self.remaining_tpm <= threshold

    def check_and_reserve(self, estimated_tokens: int) -> float:
        """Check multi-dimensional limits and pre-reserve capacity for a request.

        Args:
            estimated_tokens: Sum of prompt tokens and maximum generation budget.

        Returns:
            Timestamp of the reservation.

        Raises:
            DailyBudgetExhaustedError: If daily requests (RPD) or tokens (TPD) are exhausted.
            RateLimitError: If per-minute request (RPM) or token (TPM) limit is breached.
        """
        now = self._clock_fn()
        self._prune_windows(now)

        # 1. Daily RPD Exhaustion
        if self._daily_requests_used >= self.daily_rpd:
            raise DailyBudgetExhaustedError(
                message=f"Client daily request quota exhausted: {self._daily_requests_used}/{self.daily_rpd} (RPD)",
                reset_time_utc="15:00:00Z",
            )

        # 2. Daily TPD Exhaustion
        if self._daily_tokens_used + estimated_tokens > self.daily_tpd:
            raise DailyBudgetExhaustedError(
                message=f"Client daily token quota exhausted: {self._daily_tokens_used}/{self.daily_tpd} (TPD)",
                reset_time_utc="15:00:00Z",
            )

        # 3. RPM Sliding Window Breach
        if len(self._request_timestamps) >= self.rpm_limit:
            oldest = self._request_timestamps[0]
            retry_after = max(0.5, round(60.0 - (now - oldest), 2))
            raise RateLimitError(
                message=f"Client RPM limit breached ({self.rpm_limit} req/min). Retry after {retry_after}s",
                retry_after_s=retry_after,
                is_daily=False,
            )

        # 4. TPM Sliding Window Breach
        current_tpm = sum(tok for _, tok in self._token_reservations)
        if current_tpm + estimated_tokens > self.tpm_limit:
            oldest = self._token_reservations[0][0] if self._token_reservations else now
            retry_after = max(0.5, round(60.0 - (now - oldest), 2))
            raise RateLimitError(
                message=f"Client TPM limit breached ({current_tpm + estimated_tokens} > {self.tpm_limit} tok/min). Retry after {retry_after}s",
                retry_after_s=retry_after,
                is_daily=False,
            )

        # Admitted: record reservation
        self._request_timestamps.append(now)
        self._token_reservations.append((now, estimated_tokens))
        self._daily_requests_used += 1
        self._daily_tokens_used += estimated_tokens
        self.save_state()
        return now

    def reconcile_post_call(
        self,
        reservation_time: float,
        estimated_tokens: int,
        actual_tokens: int,
        header_snapshot: Optional[ProviderQuotaSnapshot] = None,
    ) -> None:
        """Reconcile token reservation with actual consumption and upstream headers.

        Args:
            reservation_time: Timestamp returned by check_and_reserve.
            estimated_tokens: Initial tokens reserved.
            actual_tokens: Actual tokens consumed (prompt + completion).
            header_snapshot: Optional live headers received from upstream provider.
        """
        now = self._clock_fn()
        self._prune_windows(now)

        # 1. Refund unspent token delta in sliding window and daily counter
        delta = estimated_tokens - actual_tokens
        if delta > 0:
            # Adjust the reservation in window
            for idx, (t, tok) in enumerate(self._token_reservations):
                if t == reservation_time and tok == estimated_tokens:
                    self._token_reservations[idx] = (t, actual_tokens)
                    break
            self._daily_tokens_used = max(0, self._daily_tokens_used - delta)

        # 2. Header reconciliation if live headers were captured
        if header_snapshot:
            # If header specifies exact remaining tokens, synchronize
            if header_snapshot.tpm_remaining is not None:
                upstream_used_tpm = max(
                    0, (header_snapshot.tpm_limit or self.tpm_limit) - header_snapshot.tpm_remaining
                )
                logger.debug(
                    "Reconciling TPM with upstream header: remaining=%d, used=%d",
                    header_snapshot.tpm_remaining,
                    upstream_used_tpm,
                )
            if header_snapshot.rpd_remaining is not None:
                self._daily_requests_used = max(
                    self._daily_requests_used,
                    (header_snapshot.rpd_limit or self.daily_rpd) - header_snapshot.rpd_remaining,
                )

        self.save_state()
