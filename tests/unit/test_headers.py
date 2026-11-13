"""Unit tests for Groq rate limit header and 429 payload parsing."""

import pytest

from relay.providers.base import ProviderQuotaSnapshot
from relay.providers.headers import (
    parse_duration_string,
    parse_groq_429_payload,
    parse_rate_limit_headers,
    update_snapshot_from_headers,
)


def test_parse_duration_string():
    """Verify parsing various ISO and shorthand duration strings into float seconds."""
    assert parse_duration_string("1.85") == 1.85
    assert parse_duration_string("1.85s") == 1.85
    assert parse_duration_string("450ms") == 0.45
    assert parse_duration_string("14m20s") == (14 * 60 + 20)
    assert parse_duration_string("2h15m30s") == (2 * 3600 + 15 * 60 + 30)
    assert parse_duration_string("") is None
    assert parse_duration_string(None) is None


def test_parse_rate_limit_headers():
    """Verify extraction of standard Groq/OpenAI rate limit headers."""
    headers = {
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-remaining-requests": "14380",
        "x-ratelimit-reset-requests": "14m20s",
        "x-ratelimit-limit-tokens": "6000",
        "x-ratelimit-remaining-tokens": "5850",
        "x-ratelimit-reset-tokens": "1.5s",
        "retry-after": "2.0",
    }
    parsed = parse_rate_limit_headers(headers)
    assert parsed["rpd_limit"] == 14400
    assert parsed["rpd_remaining"] == 14380
    assert parsed["rpd_reset_s"] == 860.0  # 14m20s = 860s
    assert parsed["tpm_limit"] == 6000
    assert parsed["tpm_remaining"] == 5850
    assert parsed["tpm_reset_s"] == 1.5
    assert parsed["retry_after_s"] == 2.0


def test_parse_groq_429_payload_rpm_burst():
    """Verify parsing of real Groq RPM 429 error JSON payload."""
    payload = {
        "error": {
            "message": (
                "Rate limit reached for model `llama-3.1-8b-instant` in organization `org_default` "
                "on requests per minute (RPM): Limit 30, Used 30, Requested 1. "
                "Please try again in 1.85s. Visit https://console.groq.com/docs/rate-limits"
            ),
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
    parsed = parse_groq_429_payload(payload)
    assert parsed["metric"] == "RPM"
    assert parsed["limit"] == 30
    assert parsed["used"] == 30
    assert parsed["requested"] == 1
    assert parsed["retry_after_s"] == 1.85
    assert parsed["is_daily"] is False


def test_parse_groq_429_payload_rpd_daily():
    """Verify parsing of Groq daily request budget exhaustion payload."""
    payload = {
        "error": {
            "message": (
                "Rate limit reached for model `llama-3.1-8b-instant` on requests per day (RPD): "
                "Limit 14400, Used 14400, Requested 1. Please try again in 3h15m."
            ),
            "type": "requests",
            "code": "rate_limit_exceeded",
        }
    }
    parsed = parse_groq_429_payload(payload)
    assert parsed["metric"] == "RPD"
    assert parsed["limit"] == 14400
    assert parsed["is_daily"] is True


def test_update_snapshot_from_headers():
    """Verify updating a ProviderQuotaSnapshot preserves untracked metrics."""
    initial_snapshot = ProviderQuotaSnapshot(
        rpd_limit=14400,
        rpd_remaining=14400,
        tpm_limit=6000,
        tpm_remaining=6000,
    )
    headers = {
        "x-ratelimit-remaining-requests": "14350",
        "x-ratelimit-remaining-tokens": "5200",
    }
    updated = update_snapshot_from_headers(initial_snapshot, headers)
    assert updated.rpd_limit == 14400
    assert updated.rpd_remaining == 14350
    assert updated.tpm_limit == 6000
    assert updated.tpm_remaining == 5200

