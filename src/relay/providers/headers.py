"""Header and error payload parsing for Groq API responses.

Parses x-ratelimit-* headers and HTTP 429 error bodies to extract:
- Requests Per Day (RPD) limit, remaining, reset duration
- Tokens Per Minute (TPM) limit, remaining, reset duration
- Requests Per Minute (RPM) limits and error messages
- Retry-After durations
"""

from datetime import datetime, timezone
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from relay.providers.base import ProviderQuotaSnapshot

# Regex for extracting limit metrics from Groq 429 error messages
GROQ_429_MESSAGE_PATTERN = re.compile(
    r"Rate limit reached\s*(?:for model `(?P<model>[^`]+)`.*?)?"
    r"on\s+(?P<metric_text>[^:]+?):\s*Limit\s*(?P<limit>\d+)"
    r"(?:,\s*Used\s*(?P<used>\d+))?"
    r"(?:,\s*Requested\s*(?P<requested>\d+))?",
    re.IGNORECASE,
)

# Regex for extracting retry-after duration from error message text
GROQ_RETRY_AFTER_PATTERN = re.compile(
    r"Please try again in\s*(?P<retry_after>[0-9a-zA-Z\.]+)",
    re.IGNORECASE,
)


def parse_duration_string(val: Optional[str]) -> Optional[float]:
    """Parse time string like '14m23.5s', '450ms', '2.5s', '3h15m' into float seconds."""
    if not val:
        return None

    val = str(val).strip()
    # Try pure float first
    try:
        return float(val)
    except ValueError:
        pass

    # Extract all value+unit pairs
    pattern = re.compile(
        r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>ms|msec|milliseconds?|h|hours?|m(?!s)|mins?|minutes?|s|secs?|seconds?)",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(val))
    if not matches:
        return None

    total_seconds = 0.0
    for m in matches:
        num = float(m.group("val"))
        unit = m.group("unit").lower()
        if unit.startswith("h"):
            total_seconds += num * 3600.0
        elif unit.startswith("ms"):
            total_seconds += num / 1000.0
        elif unit.startswith("m"):
            total_seconds += num * 60.0
        elif unit.startswith("s"):
            total_seconds += num

    return total_seconds if total_seconds > 0 else None


def parse_int_header(val: Optional[str]) -> Optional[int]:
    """Safely convert header value to integer."""
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def parse_rate_limit_headers(headers: Mapping[str, str]) -> Dict[str, Any]:
    """Extract standard rate limit telemetry from HTTP response headers.

    Compatible with Groq and OpenAI header conventions (case-insensitive).
    """
    # Normalize headers to lowercase
    h = {k.lower(): v for k, v in headers.items()}

    rpd_limit = parse_int_header(h.get("x-ratelimit-limit-requests"))
    rpd_remaining = parse_int_header(h.get("x-ratelimit-remaining-requests"))
    rpd_reset_s = parse_duration_string(h.get("x-ratelimit-reset-requests"))

    tpm_limit = parse_int_header(h.get("x-ratelimit-limit-tokens"))
    tpm_remaining = parse_int_header(h.get("x-ratelimit-remaining-tokens"))
    tpm_reset_s = parse_duration_string(h.get("x-ratelimit-reset-tokens"))

    # Check for direct RPM headers if present
    rpm_limit = parse_int_header(h.get("x-ratelimit-limit-rpm"))
    rpm_remaining = parse_int_header(h.get("x-ratelimit-remaining-rpm"))

    # Retry-After header
    retry_after_raw = h.get("retry-after")
    retry_after_s = parse_duration_string(retry_after_raw)

    return {
        "rpd_limit": rpd_limit,
        "rpd_remaining": rpd_remaining,
        "rpd_reset_s": rpd_reset_s,
        "tpm_limit": tpm_limit,
        "tpm_remaining": tpm_remaining,
        "tpm_reset_s": tpm_reset_s,
        "rpm_limit": rpm_limit,
        "rpm_remaining": rpm_remaining,
        "retry_after_s": retry_after_s,
    }


def parse_groq_429_payload(error_body: Any) -> Dict[str, Any]:
    """Parse Groq 429 error JSON payload or message string.

    Returns extracted limit metrics and retry-after guidance.
    """
    message = ""
    if isinstance(error_body, dict):
        # OpenAI/Groq standard structure: {"error": {"message": "...", "type": "tokens", "code": "rate_limit_exceeded"}}
        err = error_body.get("error", {})
        if isinstance(err, dict):
            message = err.get("message", "")
        else:
            message = str(err)
    elif isinstance(error_body, str):
        message = error_body

    match = GROQ_429_MESSAGE_PATTERN.search(message)
    retry_match = GROQ_RETRY_AFTER_PATTERN.search(message)
    retry_s = parse_duration_string(retry_match.group("retry_after")) if retry_match else None

    if not match:
        return {
            "message": message,
            "metric": None,
            "limit": None,
            "used": None,
            "requested": None,
            "retry_after_s": retry_s,
            "is_daily": "day" in message.lower() or "rpd" in message.lower(),
        }

    metric_str = match.group("metric_text").upper()
    if "MINUTE" in metric_str or "RPM" in metric_str:
        metric = "RPM" if "REQUEST" in metric_str or "RPM" in metric_str else "TPM"
    elif "DAY" in metric_str or "RPD" in metric_str:
        metric = "RPD" if "REQUEST" in metric_str or "RPD" in metric_str else "TPD"
    else:
        metric = metric_str

    limit = int(match.group("limit")) if match.group("limit") else None
    used = int(match.group("used")) if match.group("used") else None
    requested = int(match.group("requested")) if match.group("requested") else None

    is_daily = metric in ("RPD", "TPD")

    return {
        "message": message,
        "metric": metric,
        "limit": limit,
        "used": used,
        "requested": requested,
        "retry_after_s": retry_s,
        "is_daily": is_daily,
    }


def update_snapshot_from_headers(
    snapshot: ProviderQuotaSnapshot,
    headers: Mapping[str, str],
) -> ProviderQuotaSnapshot:
    """Update an existing ProviderQuotaSnapshot with newly received HTTP headers."""
    parsed = parse_rate_limit_headers(headers)

    rpd_limit = parsed["rpd_limit"] if parsed["rpd_limit"] is not None else snapshot.rpd_limit
    rpd_remaining = parsed["rpd_remaining"] if parsed["rpd_remaining"] is not None else snapshot.rpd_remaining
    rpd_reset_s = parsed["rpd_reset_s"] if parsed["rpd_reset_s"] is not None else snapshot.rpd_reset_s

    tpm_limit = parsed["tpm_limit"] if parsed["tpm_limit"] is not None else snapshot.tpm_limit
    tpm_remaining = parsed["tpm_remaining"] if parsed["tpm_remaining"] is not None else snapshot.tpm_remaining
    tpm_reset_s = parsed["tpm_reset_s"] if parsed["tpm_reset_s"] is not None else snapshot.tpm_reset_s

    rpm_limit = parsed["rpm_limit"] if parsed["rpm_limit"] is not None else snapshot.rpm_limit
    rpm_remaining = parsed["rpm_remaining"] if parsed["rpm_remaining"] is not None else snapshot.rpm_remaining

    return ProviderQuotaSnapshot(
        rpd_limit=rpd_limit,
        rpd_remaining=rpd_remaining,
        rpd_reset_s=rpd_reset_s,
        tpm_limit=tpm_limit,
        tpm_remaining=tpm_remaining,
        tpm_reset_s=tpm_reset_s,
        rpm_limit=rpm_limit,
        rpm_remaining=rpm_remaining,
        last_updated_iso=datetime.now(timezone.utc).isoformat(),
    )
