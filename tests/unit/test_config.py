"""Unit tests for Phase 1 configuration loading, SecretStr masking, and scenario specs."""

from pathlib import Path
import pytest
import yaml
from pydantic import SecretStr

from relay.config.settings import RelayConfig


def test_load_default_yaml() -> None:
    """Verify loading default.yaml produces a valid RelayConfig object."""
    config = RelayConfig.load_from_yaml(yaml_path=Path("config/default.yaml"))
    assert config.provider.model_id == "llama-3.1-8b-instant"
    assert config.provider.request_timeout_s == 15.0
    assert config.provider.max_retries == 2
    assert config.standin.rate_limit_rpm == 30
    assert config.reliability.max_queue_size == 24
    assert config.reliability.compact_pieces_k == 2
    assert config.rag.default_setting == "S3_450_8to4_rrf_no"
    assert config.observability.results_csv_path == "results.csv"


def test_secret_str_masking() -> None:
    """Verify that GROQ_API_KEY is masked by SecretStr and never leaks in repr/str."""
    config = RelayConfig(
        provider={"api_key": SecretStr("gsk_super_secret_test_key_1234567890")}  # type: ignore[arg-type]
    )
    assert config.provider.api_key is not None
    # SecretStr repr/str must NOT contain the plain key
    assert "gsk_super_secret_test_key_1234567890" not in str(config)
    assert "gsk_super_secret_test_key_1234567890" not in repr(config)
    assert "**********" in str(config.provider.api_key)
    # The actual secret is accessible explicitly via get_secret_value()
    assert config.provider.api_key.get_secret_value() == "gsk_super_secret_test_key_1234567890"


def test_config_numerical_bounds() -> None:
    """Verify Pydantic enforces valid bounds on configuration parameters."""
    with pytest.raises(Exception):
        # max_queue_size cannot be 0 or negative
        RelayConfig(reliability={"max_queue_size": 0})  # type: ignore[arg-type]

    with pytest.raises(Exception):
        # tpm_compact_threshold_pct must be between 0.05 and 0.50
        RelayConfig(reliability={"tpm_compact_threshold_pct": 0.90})  # type: ignore[arg-type]


def test_scenarios_yaml_structure() -> None:
    """Verify config/scenarios.yaml contains all 6 required benchmark shapes."""
    scenario_file = Path("config/scenarios.yaml")
    assert scenario_file.is_file(), "config/scenarios.yaml must exist"

    with open(scenario_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "scenarios" in data
    scenarios = data["scenarios"]

    expected_scenarios = [
        "steady_real",
        "steady_standin",
        "burst",
        "sustained_overload",
        "rpm_halved",
        "daily_budget_exhausted",
    ]

    for sc_name in expected_scenarios:
        assert sc_name in scenarios, f"Missing scenario {sc_name}"
        sc = scenarios[sc_name]
        assert "scenario" in sc
        assert "source" in sc
        assert "num_requests" in sc
        assert "duration_s" in sc
        assert "offered_rate_rpm" in sc
        assert "concurrency_cap" in sc
        assert sc["num_requests"] > 0
        assert sc["duration_s"] > 0
        assert sc["concurrency_cap"] > 0


def test_config_nonexistent_yaml_fallback() -> None:
    """Verify loading from non-existent file path falls back to default values cleanly."""
    cfg = RelayConfig.load_from_yaml(yaml_path=Path("non_existent_file.yaml"))
    assert cfg.provider.model_id == "llama-3.1-8b-instant"
    assert cfg.reliability.max_queue_size == 24

