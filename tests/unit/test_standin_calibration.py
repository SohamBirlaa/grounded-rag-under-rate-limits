"""Unit tests for Phase 4: Controllable & Calibrated Stand-In Model Service.

Verifies:
1. Parametric TTFT behavior (prompt scaling, noise, slow TTFT mode)
2. ITL streaming cadence, chunking, and mock sleep_fn support
3. Concurrency degradation (knee threshold, slope multiplier, active tracking)
4. Multi-dimensional rate limit tracking (RPM, TPM, RPD, TPD)
5. Timeout simulation and error distinction
6. Deterministic grounded response reproducibility
7. Epistemic abstention on missing passages
8. Failure injection modes and fail_attempts counters
9. Calibration artifact schema, provenance, and zero-live-call disclosures
10. ModelProviderPort protocol conformance and provider factory integration
11. ReliabilityLoop integration and retry recovery
"""

import asyncio
import json
from pathlib import Path
import pytest

from relay.config.settings import RelayConfig, StandInConfig
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
from relay.providers.factory import get_provider
from relay.providers.standin import StandInProviderAdapter, StandInRateTracker
from relay.reliability.retry import ReliabilityLoop, RetryPolicy


@pytest.fixture
def sample_rag_prompt():
    return (
        "Here are the relevant passages:\n\n"
        '<passage id="1706.03762v7_sec_3_2_p04_c014" paper_id="1706.03762v7" '
        'version="v7" page="4" section="3.2.1 Scaled Dot-Product Attention" content_type="prose">\n'
        "We compute the attention function on a set of queries simultaneously, "
        "packed together into a matrix Q.\n"
        "</passage>\n\n"
        "Question: What is Scaled Dot-Product Attention?"
    )


# --- 1. Parametric TTFT Tests ---

def test_parametric_ttft_prompt_length_scaling():
    """Verify TTFT scales linearly with prompt tokens: base + slope * prompt_tokens."""
    cfg = StandInConfig(
        base_ttft_ms=280.0,
        prompt_token_slope=0.05,
        noise_sigma_ms=0.0,  # Zero noise for deterministic calculation
        concurrency_knee=4,
    )
    standin = StandInProviderAdapter(config=cfg, inject_delay=False)

    ttft_0 = standin.calculate_simulated_ttft_ms(prompt_tokens=0, active_concurrency=1)
    ttft_100 = standin.calculate_simulated_ttft_ms(prompt_tokens=100, active_concurrency=1)
    ttft_500 = standin.calculate_simulated_ttft_ms(prompt_tokens=500, active_concurrency=1)

    assert ttft_0 == 280.0
    assert ttft_100 == 280.0 + (0.05 * 100) == 285.0
    assert ttft_500 == 280.0 + (0.05 * 500) == 305.0


def test_parametric_ttft_slow_ttft_mode():
    """Verify failure_mode='slow_ttft' overrides calculated TTFT with configured stall delay."""
    cfg = StandInConfig(base_ttft_ms=280.0, slow_ttft_ms=3500.0)
    standin = StandInProviderAdapter(
        config=cfg,
        inject_delay=False,
        failure_mode="slow_ttft",
    )
    assert standin.calculate_simulated_ttft_ms(prompt_tokens=50) == 3500.0


def test_parametric_ttft_deterministic_noise():
    """Verify seeded Gaussian noise is reproducible across instances."""
    cfg = StandInConfig(base_ttft_ms=280.0, prompt_token_slope=0.0, noise_sigma_ms=15.0)
    standin1 = StandInProviderAdapter(config=cfg, inject_delay=True, random_seed=123)
    standin2 = StandInProviderAdapter(config=cfg, inject_delay=True, random_seed=123)

    val1_seq = [standin1.calculate_simulated_ttft_ms(prompt_tokens=0) for _ in range(5)]
    val2_seq = [standin2.calculate_simulated_ttft_ms(prompt_tokens=0) for _ in range(5)]

    assert val1_seq == val2_seq
    assert any(v != 280.0 for v in val1_seq)


# --- 2. Inter-Token Latency (ITL) & Pacing Tests ---

def test_itl_cadence_and_slow_tpot_mode():
    """Verify base ITL vs slow TPOT mode."""
    cfg = StandInConfig(itl_ms=4.5, slow_tpot_ms=60.0, concurrency_knee=4)
    standin_normal = StandInProviderAdapter(config=cfg, inject_delay=False)
    standin_slow = StandInProviderAdapter(
        config=cfg,
        inject_delay=False,
        failure_mode="slow_tpot",
    )

    assert standin_normal.calculate_simulated_itl_ms(active_concurrency=1) == 4.5
    assert standin_slow.calculate_simulated_itl_ms(active_concurrency=1) == 60.0


def test_custom_sleep_fn_receives_delays(sample_rag_prompt):
    """Verify custom sleep_fn captures simulated TTFT and ITL delays without actual blocking."""
    recorded_delays = []

    async def mock_sleep(s: float) -> None:
        recorded_delays.append(s)

    cfg = StandInConfig(base_ttft_ms=100.0, itl_ms=5.0, chunk_size=3)
    standin = StandInProviderAdapter(
        config=cfg,
        inject_delay=True,
        sleep_fn=mock_sleep,
    )

    req = ProviderCompletionRequest(
        messages=[{"role": "user", "content": sample_rag_prompt}],
    )

    async def _run():
        chunks = []
        async for c in standin.stream_completion(req):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_run())
    assert len(chunks) > 0
    # First sleep is TTFT delay
    assert len(recorded_delays) >= 2
    assert recorded_delays[0] >= 0.100  # >= 100ms TTFT
    # Subsequent sleeps are chunk ITL delays
    assert all(d > 0.0 for d in recorded_delays[1:])


# --- 3. Concurrency Degradation Tests ---

def test_concurrency_degradation_factor_calculation():
    """Verify mathematical degradation factor curve: 1.0 at <= knee, +slope*(C-knee) above."""
    cfg = StandInConfig(concurrency_knee=4, concurrency_slope=0.15)
    standin = StandInProviderAdapter(config=cfg, inject_delay=False)

    assert standin.get_concurrency_degradation(active_count=1) == 1.0
    assert standin.get_concurrency_degradation(active_count=4) == 1.0
    # C=8: 1.0 + 0.15 * (8 - 4) = 1.60
    assert pytest.approx(standin.get_concurrency_degradation(active_count=8)) == 1.60
    # C=12: 1.0 + 0.15 * (12 - 4) = 2.20
    assert pytest.approx(standin.get_concurrency_degradation(active_count=12)) == 2.20


def test_concurrency_degradation_scales_ttft_and_itl():
    """Verify TTFT and ITL are scaled by concurrency degradation."""
    cfg = StandInConfig(
        base_ttft_ms=200.0,
        prompt_token_slope=0.0,
        noise_sigma_ms=0.0,
        itl_ms=5.0,
        concurrency_knee=4,
        concurrency_slope=0.20,
    )
    standin = StandInProviderAdapter(config=cfg, inject_delay=False)

    # Under C=4 (no degradation)
    ttft_4 = standin.calculate_simulated_ttft_ms(prompt_tokens=0, active_concurrency=4)
    itl_4 = standin.calculate_simulated_itl_ms(active_concurrency=4)
    assert ttft_4 == 200.0
    assert itl_4 == 5.0

    # Under C=9 (knee=4, excess=5 -> factor = 1.0 + 0.20*5 = 2.0)
    ttft_9 = standin.calculate_simulated_ttft_ms(prompt_tokens=0, active_concurrency=9)
    itl_9 = standin.calculate_simulated_itl_ms(active_concurrency=9)
    assert ttft_9 == 400.0
    assert itl_9 == 10.0


# --- 4. Multi-Dimensional Rate Limit Simulation Tests ---

def test_rate_tracker_rpm_sliding_window_breach():
    """Verify rate tracker raises RateLimitError when 60s requests exceed RPM limit."""
    simulated_time = 1000.0

    def clock():
        return simulated_time

    tracker = StandInRateTracker(rpm_limit=3, clock_fn=clock)

    # First 3 requests admitted
    tracker.check_limits(estimated_tokens=50)
    tracker.record_request_admitted(estimated_tokens=50)

    tracker.check_limits(estimated_tokens=50)
    tracker.record_request_admitted(estimated_tokens=50)

    tracker.check_limits(estimated_tokens=50)
    tracker.record_request_admitted(estimated_tokens=50)

    # 4th request within same window triggers 429
    with pytest.raises(RateLimitError) as exc_info:
        tracker.check_limits(estimated_tokens=50)

    assert exc_info.value.limit_type == "RPM"
    assert exc_info.value.is_daily is False
    assert exc_info.value.retry_after_s > 0

    # Advance clock past 60s -> window clears
    simulated_time += 61.0
    tracker.check_limits(estimated_tokens=50)  # Succeeds without error


def test_rate_tracker_tpm_limit_breach():
    """Verify rate tracker raises RateLimitError when 60s tokens exceed TPM limit."""
    simulated_time = 2000.0

    def clock():
        return simulated_time

    tracker = StandInRateTracker(tpm_limit=500, clock_fn=clock)

    tracker.record_request_admitted(estimated_tokens=400)

    # Next request with 150 tokens exceeds 500 TPM limit
    with pytest.raises(RateLimitError) as exc_info:
        tracker.check_limits(estimated_tokens=150)

    assert exc_info.value.limit_type == "TPM"
    assert exc_info.value.is_daily is False


def test_rate_tracker_daily_quota_exhaustion():
    """Verify daily request (RPD) and daily token (TPD) limits trigger DailyBudgetExhaustedError."""
    tracker_rpd = StandInRateTracker(daily_rpd=2)
    tracker_rpd.record_request_admitted(10)
    tracker_rpd.record_request_admitted(10)

    with pytest.raises(DailyBudgetExhaustedError):
        tracker_rpd.check_limits(10)

    tracker_tpd = StandInRateTracker(daily_tpd=1000)
    tracker_tpd.record_request_admitted(900)

    with pytest.raises(DailyBudgetExhaustedError):
        tracker_tpd.check_limits(150)


def test_standin_enforces_rate_limits_when_enabled(sample_rag_prompt):
    """Verify StandInProviderAdapter raises 429 when enforce_rate_limits=True and RPM is breached."""
    cfg = StandInConfig(rate_limit_rpm=2)
    standin = StandInProviderAdapter(
        config=cfg,
        inject_delay=False,
        enforce_rate_limits=True,
    )

    req = ProviderCompletionRequest(
        messages=[{"role": "user", "content": sample_rag_prompt}],
    )

    async def _run():
        # Call 1: OK
        async for _ in standin.stream_completion(req):
            pass
        # Call 2: OK
        async for _ in standin.stream_completion(req):
            pass
        # Call 3: Should trigger RPM limit
        async for _ in standin.stream_completion(req):
            pass

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.limit_type == "RPM"


# --- 5. Timeout & Failure Injection Tests ---

def test_standin_all_failure_modes(sample_rag_prompt):
    """Verify all failure modes map to their respective exception types."""
    req = ProviderCompletionRequest(
        messages=[{"role": "user", "content": sample_rag_prompt}],
    )

    modes_to_errors = [
        ("auth_error", ProviderAuthenticationError),
        ("rate_limit_burst", RateLimitError),
        ("rate_limit_tpm", RateLimitError),
        ("daily_exhausted", DailyBudgetExhaustedError),
        ("server_error", ProviderUnavailableError),
        ("transient_network", ProviderUnavailableError),
        ("timeout", ProviderTimeoutError),
    ]

    for mode, expected_error in modes_to_errors:
        standin = StandInProviderAdapter(
            inject_delay=False,
            failure_mode=mode,
        )

        async def _test():
            async for _ in standin.stream_completion(req):
                pass

        with pytest.raises(expected_error):
            asyncio.run(_test())


# --- 6. Deterministic Response Reproducibility Tests ---

def test_deterministic_response_reproducibility(sample_rag_prompt):
    """Verify identical prompts yield byte-for-byte identical output sequences and citations."""
    standin1 = StandInProviderAdapter(inject_delay=False)
    standin2 = StandInProviderAdapter(inject_delay=False)

    req = ProviderCompletionRequest(
        messages=[{"role": "user", "content": sample_rag_prompt}],
    )

    async def _get_output(provider):
        chunks = []
        async for c in provider.stream_completion(req):
            chunks.append(c)
        return "".join(c.text_delta for c in chunks)

    out1 = asyncio.run(_get_output(standin1))
    out2 = asyncio.run(_get_output(standin2))

    assert out1 == out2
    assert "arXiv:1706.03762v7" in out1
    assert "Scaled Dot-Product Attention" in out1 or "attention function" in out1


# --- 7. Calibration Schema & Provenance Validation ---

def test_calibration_json_schema_and_provenance():
    """Verify config/calibration.json is complete, valid, and declares 0 live calls."""
    calib_file = Path("config/calibration.json")
    assert calib_file.is_file(), "config/calibration.json must exist"

    with open(calib_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "parameters" in data
    assert "residuals_evaluation" in data
    assert "differences_from_real_service" in data

    # Zero live call declaration
    meta = data["metadata"]
    assert meta["live_groq_api_calls_in_phase_3"] == 0
    assert meta["live_groq_api_calls_in_phase_4"] == 0
    assert meta["external_quota_consumed_tokens"] == 0

    # Required calibration parameters inventory
    params = data["parameters"]
    required_keys = [
        "base_ttft_ms",
        "prompt_token_slope",
        "noise_sigma_ms",
        "slow_ttft_ms",
        "itl_ms",
        "slow_tpot_ms",
        "chunk_size",
        "concurrency_knee",
        "concurrency_slope",
        "in_flight_concurrency_cap",
        "rate_limit_rpm",
        "rate_limit_tpm",
        "daily_rpd",
        "daily_tpd",
    ]
    for k in required_keys:
        assert k in params, f"Missing parameter {k} in config/calibration.json"
        p = params[k]
        assert "value" in p
        assert "unit" in p
        assert "provenance" in p
        assert "description" in p

    # Confirm residuals are explicitly marked unavailable
    residuals = data["residuals_evaluation"]
    assert residuals["status"] == "UNAVAILABLE_ZERO_LIVE_CALLS"
    assert residuals["real_measurements"] is None


# --- 8. Provider Interface Compatibility & Factory Integration ---

def test_standin_implements_model_provider_port():
    """Verify StandInProviderAdapter structurally satisfies ModelProviderPort protocol."""
    standin = StandInProviderAdapter(inject_delay=False)
    # Check required protocol methods
    assert hasattr(standin, "stream_completion")
    assert callable(standin.stream_completion)
    assert hasattr(standin, "get_quota_snapshot")
    assert callable(standin.get_quota_snapshot)
    assert hasattr(standin, "check_health")
    assert callable(standin.check_health)


def test_factory_standin_integration():
    """Verify get_provider strictly returns StandInProviderAdapter when source='stand-in'."""
    cfg = RelayConfig()
    provider = get_provider(config=cfg, source="stand-in", inject_delay=False)
    assert isinstance(provider, StandInProviderAdapter)
    assert provider.inject_delay is False


# --- 9. ReliabilityLoop Integration & Recovery Tests ---

def test_standin_with_reliability_loop_recovery(sample_rag_prompt):
    """Verify ReliabilityLoop recovers from transient stand-in error after attempt fails."""
    standin = StandInProviderAdapter(
        inject_delay=False,
        failure_mode="server_error",
        fail_attempts=1,  # Fails attempt 1, succeeds attempt 2
    )
    policy = RetryPolicy(max_retries=2, base_backoff_s=0.01)

    async def mock_sleep(s):
        pass

    loop = ReliabilityLoop(retry_policy=policy, sleep_fn=mock_sleep)
    req = ProviderCompletionRequest(
        messages=[{"role": "user", "content": sample_rag_prompt}],
    )

    async def _run():
        return await loop.execute_stream(provider=standin, request=req)

    result = asyncio.run(_run())
    assert result.success is True
    assert result.attempts == 2
    assert result.retries == 1
    assert "arXiv:1706.03762v7" in result.assembled_text
