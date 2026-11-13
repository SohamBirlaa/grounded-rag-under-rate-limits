"""Rate-limit bookkeeping driven by Groq's real response headers.

Groq headers (documented): x-ratelimit-limit-requests / remaining-requests /
reset-requests refer to requests PER DAY; x-ratelimit-*-tokens refer to tokens
PER MINUTE; retry-after is sent on 429. We combine those provider readings with
a local sliding window, because headers are only seen after a response.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

_DUR_RE = re.compile(r"(?P<val>\d+(?:\.\d+)?)(?P<unit>ms|h|m|s)")


def parse_duration(value: Optional[str]) -> Optional[float]:
    """Parse Groq duration strings ('7.66s', '2m59.5s', '3h15m', '120ms', '1.5') to seconds."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    total, matched = 0.0, False
    for m in _DUR_RE.finditer(s):
        matched = True
        v = float(m.group("val"))
        unit = m.group("unit")
        total += {"ms": v / 1000.0, "s": v, "m": v * 60.0, "h": v * 3600.0}[unit]
    return total if matched else None


def _int(v: Optional[str]) -> Optional[int]:
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class HeaderSnapshot:
    observed_at: float
    rpd_limit: Optional[int] = None
    rpd_remaining: Optional[int] = None
    rpd_reset_s: Optional[float] = None
    tpm_limit: Optional[int] = None
    tpm_remaining: Optional[int] = None
    tpm_reset_s: Optional[float] = None
    retry_after_s: Optional[float] = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str], now: float) -> "HeaderSnapshot":
        h = {k.lower(): v for k, v in headers.items()}
        return cls(
            observed_at=now,
            rpd_limit=_int(h.get("x-ratelimit-limit-requests")),
            rpd_remaining=_int(h.get("x-ratelimit-remaining-requests")),
            rpd_reset_s=parse_duration(h.get("x-ratelimit-reset-requests")),
            tpm_limit=_int(h.get("x-ratelimit-limit-tokens")),
            tpm_remaining=_int(h.get("x-ratelimit-remaining-tokens")),
            tpm_reset_s=parse_duration(h.get("x-ratelimit-reset-tokens")),
            retry_after_s=parse_duration(h.get("retry-after")),
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_RETRY_IN_RE = re.compile(r"try again in ([0-9hms\.]+)", re.I)


def classify_429(message: str) -> Tuple[str, Optional[float]]:
    """Return ('day'|'minute', retry_after_s parsed from the message)."""
    msg = message or ""
    low = msg.lower()
    kind = "day" if ("per day" in low or "(rpd)" in low or "(tpd)" in low) else "minute"
    m = _RETRY_IN_RE.search(msg)
    return kind, (parse_duration(m.group(1)) if m else None)


class TokenBudget:
    """Admission control for one Groq model (limits are per model).

    Measured on the real service (live_probe): Groq's per-minute token limit behaves as a
    continuously refilling bucket -- `x-ratelimit-reset-tokens` equals tokens_used / (TPM/60)
    (1,655 used -> 12.41 s at 8K TPM), and the daily request limit refills one request every
    86,400 / RPD seconds (86.4 s at 1K/day). So we model both minute limits as buckets refilling
    at limit/60 per second instead of a fixed 60 s window, which would over-wait after a burst.

    `safety` keeps us below the provider ceiling so our pacing, not a 429, delays a request.
    Provider headers can only LOWER our estimate (other in-flight requests may not be counted yet).
    """

    def __init__(
        self,
        rpm: int,
        tpm: int,
        rpd: Optional[int] = None,
        safety: float = 0.9,
        local_daily_request_cap: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rpm, self.tpm, self.rpd = rpm, tpm, rpd
        self.safety = safety
        self.local_daily_request_cap = local_daily_request_cap
        self.clock = clock
        now = clock()
        self._tok_level, self._req_level, self._t = self.tok_cap, self.req_cap, now
        self.requests_today = 0
        self.tokens_today = 0
        self.daily_exhausted = False
        self.daily_reset_s: Optional[float] = None
        self.daily_exhausted_at: Optional[float] = None
        self.last_snapshot: Optional[HeaderSnapshot] = None
        self._provider_block_until = 0.0

    # ---------- bucket mechanics ----------
    @property
    def tok_cap(self) -> float:
        return self.tpm * self.safety

    @property
    def req_cap(self) -> float:
        return max(1.0, self.rpm * self.safety)

    def _refill(self, now: Optional[float] = None) -> float:
        now = self.clock() if now is None else now
        dt = max(0.0, now - self._t)
        self._tok_level = min(self.tok_cap, self._tok_level + dt * self.tpm / 60.0)
        self._req_level = min(self.req_cap, self._req_level + dt * self.rpm / 60.0)
        self._t = now
        return now

    def available_tokens(self) -> float:
        self._refill()
        return self._tok_level

    def tokens_in_window(self, now: Optional[float] = None) -> int:
        """Tokens currently committed (capacity minus what is available)."""
        self._refill(now)
        return int(self.tok_cap - self._tok_level)

    def pressure(self) -> float:
        """Fraction of the per-minute token bucket in use (0 = idle, 1 = empty bucket)."""
        self._refill()
        return max(0.0, 1.0 - self._tok_level / self.tok_cap)

    # ---------- provider feedback ----------
    def observe(self, snap: HeaderSnapshot) -> None:
        self.last_snapshot = snap
        if snap.rpd_remaining is not None and snap.rpd_remaining <= 0:
            self.mark_daily_exhausted(snap.rpd_reset_s)
        if snap.tpm_remaining is not None:
            self._refill()
            margin = (snap.tpm_limit or self.tpm) - self.tok_cap
            self._tok_level = min(self._tok_level, max(0.0, snap.tpm_remaining - margin))

    def observe_429(self, kind: str, retry_after_s: Optional[float]) -> None:
        now = self.clock()
        if kind == "day":
            self.mark_daily_exhausted(retry_after_s)
        else:
            self._refill(now)
            self._tok_level = min(self._tok_level, 0.0)
            if retry_after_s:
                self._provider_block_until = max(self._provider_block_until, now + retry_after_s)

    def mark_daily_exhausted(self, reset_s: Optional[float]) -> None:
        self.daily_exhausted = True
        self.daily_exhausted_at = self.clock()
        if reset_s is not None:
            self.daily_reset_s = reset_s

    # ---------- admission ----------
    def wait_time(self, est_tokens: int, now: Optional[float] = None) -> float:
        """Seconds until a request of `est_tokens` may be sent (0 = now, inf = never fits)."""
        now = self._refill(now)
        if est_tokens > self.tok_cap:
            return float("inf")  # caller must shrink the prompt
        waits = [max(0.0, self._provider_block_until - now)]
        if self._tok_level < est_tokens:
            waits.append((est_tokens - self._tok_level) / (self.tpm / 60.0))
        if self._req_level < 1.0:
            waits.append((1.0 - self._req_level) / (self.rpm / 60.0))
        return max(waits)

    def daily_blocked(self) -> bool:
        if self.daily_exhausted:
            # Measured: Groq refills the daily token budget continuously (200K/86,400 s = 2.31 tok/s; a 969-token
            # shortfall -> "try again in 6m58.6s"). So re-open once its retry-after has passed instead of staying
            # closed all day. (A local test cap has no reset time and stays closed.)
            if (self.daily_reset_s is not None and self.daily_exhausted_at is not None
                    and self.clock() >= self.daily_exhausted_at + self.daily_reset_s):
                self.daily_exhausted = False
                self.daily_exhausted_at = None
            else:
                return True
        if self.local_daily_request_cap is not None and self.requests_today >= self.local_daily_request_cap:
            self.mark_daily_exhausted(None)
            return True
        return False

    def reserve(self, est_tokens: int) -> list:
        self._refill()
        self._tok_level -= est_tokens
        self._req_level -= 1.0
        self.requests_today += 1
        return [self._t, int(est_tokens)]

    def settle(self, entry: list, actual_tokens: int) -> None:
        """Replace the estimate with real usage: refund (or charge) the difference."""
        self.tokens_today += int(actual_tokens)
        self._refill()
        self._tok_level = min(self.tok_cap, self._tok_level + (entry[1] - int(actual_tokens)))
        entry[1] = int(actual_tokens)

    async def acquire(self, est_tokens: int, deadline_at: float) -> Tuple[bool, float, Optional[list]]:
        """Wait (if the deadline allows) until the request fits. Returns (admitted, waited_s, entry)."""
        start = self.clock()
        while True:
            if self.daily_blocked():
                return False, self.clock() - start, None
            w = self.wait_time(est_tokens)
            if w <= 0:
                return True, self.clock() - start, self.reserve(est_tokens)
            if self.clock() + w > deadline_at:
                return False, self.clock() - start, None  # shed now instead of making them wait for nothing
            await asyncio.sleep(min(w, 1.0))
