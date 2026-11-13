"""Unit tests verifying calibration artifact provenance and schema compliance."""

import json
from pathlib import Path
import pytest


def test_groq_429_sample_provenance():
    """Verify groq_429_sample.json contains explicit provenance metadata and no false live claims."""
    path = Path("data/calibration/groq_429_sample.json")
    assert path.is_file(), f"Missing calibration artifact: {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "artifact_metadata" in data
    meta = data["artifact_metadata"]
    assert meta["is_live_capture"] is False
    assert "Curated" in meta["provenance_classification"]
    assert "https://console.groq.com/docs/rate-limits" in meta["source_reference"]

    samples = data.get("samples", {})
    assert "burst_rpm_429" in samples
    assert "daily_rpd_429" in samples
    assert samples["burst_rpm_429"]["status_code"] == 429
    assert samples["daily_rpd_429"]["status_code"] == 429


def test_table1_limits_provenance_and_non_empirical_labeling():
    """Verify Table 1 limits properly categorize documented limits vs unverified empirical claims."""
    path = Path("data/calibration/table1_empirical_limits.json")
    assert path.is_file(), f"Missing calibration artifact: {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "limits" in data
    limits = data["limits"]

    # 1. RPM, TPM, RPD, TPD must be Documented Limits
    for metric_key in ("requests_per_minute_rpm", "tokens_per_minute_tpm", "requests_per_day_rpd", "tokens_per_day_tpd"):
        entry = limits[metric_key]
        assert entry["is_empirical_measurement"] is False
        assert "Official Provider Documentation" in entry["source_type"]
        assert "Verified against Official Provider Documentation" in entry["verification_status"]

    # 2. In-flight concurrency must NOT be labeled as empirical measurement from Phase 3
    concurrency_entry = limits["in_flight_concurrency"]
    assert concurrency_entry["is_empirical_measurement"] is False
    assert "System Architecture Design Parameter" in concurrency_entry["source_type"]
    assert "Unverified Empirical Measurement" in concurrency_entry["verification_status"]
    assert "Little's Law" in concurrency_entry["methodology_and_evidence"]

    # 3. Provenance summary must declare 0 live API calls in Phase 3
    assert data["provenance_summary"]["live_api_calls_in_phase_3"] == 0
    assert data["provenance_summary"]["quota_consumed_in_phase_3"] == 0

