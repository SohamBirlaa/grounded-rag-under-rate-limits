"""Model provider port abstraction and data models.

Defines the common asynchronous interface implemented by both the real Groq client
and the calibrated stand-in simulator.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field


class ProviderCompletionRequest(BaseModel):
    """Standardized completion request sent to an inference provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: List[Dict[str, str]] = Field(..., min_length=1)
    model_id: str = Field(default="llama-3.1-8b-instant")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = Field(default=True)
    request_id: Optional[str] = Field(default=None)
    timeout_s: float = Field(default=15.0, ge=1.0, le=120.0)


@dataclass
class ProviderStreamChunk:
    """A streaming chunk emitted during token generation."""

    text_delta: str
    is_first_token: bool
    token_index: int
    arrival_monotonic_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


@dataclass
class ProviderQuotaSnapshot:
    """Current snapshot of provider rate limits, remaining quotas, and reset times."""

    rpd_limit: Optional[int] = None
    rpd_remaining: Optional[int] = None
    rpd_reset_s: Optional[float] = None
    tpm_limit: Optional[int] = None
    tpm_remaining: Optional[int] = None
    tpm_reset_s: Optional[float] = None
    rpm_limit: Optional[int] = None
    rpm_remaining: Optional[int] = None
    last_updated_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def is_daily_exhausted(self) -> bool:
        """Check if daily request budget is completely consumed."""
        return self.rpd_remaining is not None and self.rpd_remaining <= 0

    def is_tpm_depleted(self, threshold_pct: float = 0.25) -> bool:
        """Check if remaining tokens-per-minute is below threshold for compact mode."""
        if self.tpm_limit is not None and self.tpm_remaining is not None:
            if self.tpm_limit > 0:
                return (self.tpm_remaining / self.tpm_limit) <= threshold_pct
        return False


@runtime_checkable
class ModelProviderPort(Protocol):
    """Protocol implemented by both live Groq adapter and calibrated stand-in."""

    async def stream_completion(
        self, request: ProviderCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        """Stream generated completion chunks with first-token timing."""
        ...

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        """Return the latest tracked rate-limit quota snapshot."""
        ...

    async def check_health(self) -> bool:
        """Verify provider connectivity and availability."""
        ...
