"""Inference providers package for The Relay.

Provides:
- ModelProviderPort Protocol
- ProviderCompletionRequest, ProviderStreamChunk, ProviderQuotaSnapshot
- GroqProviderAdapter
- Rate limit headers parsing utilities
"""

from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
    ProviderQuotaSnapshot,
    ProviderStreamChunk,
)
from relay.providers.headers import (
    parse_duration_string,
    parse_groq_429_payload,
    parse_rate_limit_headers,
    update_snapshot_from_headers,
)
from relay.providers.groq import GroqProviderAdapter
from relay.providers.standin import StandInProviderAdapter
from relay.providers.factory import get_provider

__all__ = [
    "ModelProviderPort",
    "ProviderCompletionRequest",
    "ProviderQuotaSnapshot",
    "ProviderStreamChunk",
    "GroqProviderAdapter",
    "StandInProviderAdapter",
    "get_provider",
    "parse_duration_string",
    "parse_groq_429_payload",
    "parse_rate_limit_headers",
    "update_snapshot_from_headers",
]
