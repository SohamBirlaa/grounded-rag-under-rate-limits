"""Typed Pydantic configuration loader for The Relay.

Loads baseline parameters from YAML and environment variables safely.
Enforces SecretStr masking on API keys to prevent accidental leakage in logs.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from relay.contracts.schemas import ProviderSourceType, RetrievalSettingType


class ProviderConfig(BaseModel):
    """Configuration for Groq or external LLM provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Groq API key loaded from GROQ_API_KEY env var (masked)",
    )
    base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for Groq API endpoint",
    )
    model_id: str = Field(
        default="llama-3.1-8b-instant",
        description="Active model identifier on Groq free tier",
    )
    request_timeout_s: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        description="Per-attempt network and streaming chunk timeout",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts on retryable transient faults",
    )
    default_source: ProviderSourceType = Field(
        default="real",
        description="Default backend source ('real' or 'stand-in')",
    )


class StandInConfig(BaseModel):
    """Configuration for local calibrated stand-in service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_ttft_ms: float = Field(
        default=280.0,
        ge=10.0,
        le=5000.0,
        description="Base Time-to-First-Token in milliseconds (provisional intercept)",
    )
    prompt_token_slope: float = Field(
        default=0.05,
        ge=0.0,
        le=10.0,
        description="Additional TTFT delay per prompt token in milliseconds",
    )
    noise_sigma_ms: float = Field(
        default=15.0,
        ge=0.0,
        le=500.0,
        description="Standard deviation of Gaussian TTFT noise in milliseconds",
    )
    slow_ttft_ms: float = Field(
        default=3500.0,
        ge=500.0,
        le=30000.0,
        description="Forced TTFT stall duration for slow_ttft mode in milliseconds",
    )
    itl_ms: float = Field(
        default=4.5,
        ge=0.5,
        le=100.0,
        description="Base Inter-Token Latency in milliseconds (~222 tokens/sec)",
    )
    slow_tpot_ms: float = Field(
        default=60.0,
        ge=1.0,
        le=1000.0,
        description="Forced decode delay per token for slow_tpot mode in milliseconds",
    )
    chunk_size: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of words/tokens per streamed SSE chunk",
    )
    load_factor: float = Field(
        default=12.0,
        ge=0.0,
        le=100.0,
        description="Legacy load factor for backwards compatibility",
    )
    concurrency_knee: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Active concurrency threshold before load degradation begins",
    )
    concurrency_slope: float = Field(
        default=0.15,
        ge=0.0,
        le=5.0,
        description="Latency degradation multiplier per active request exceeding knee",
    )
    failure_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Baseline random failure injection rate",
    )
    rate_limit_rpm: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="Documented requests per minute rate limit ceiling",
    )
    rate_limit_tpm: int = Field(
        default=6000,
        ge=100,
        le=100000,
        description="Documented tokens per minute rate limit ceiling",
    )
    daily_rpd: int = Field(
        default=14400,
        ge=1,
        le=1000000,
        description="Documented daily requests quota ceiling",
    )
    daily_tpd: int = Field(
        default=500000,
        ge=1000,
        le=10000000,
        description="Documented daily tokens quota ceiling",
    )


class ReliabilityConfig(BaseModel):
    """Admission control, queueing, and degradation thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_queue_size: int = Field(default=24, ge=1, le=256)
    queue_timeout_ms: float = Field(default=4500.0, ge=100.0, le=60000.0)
    default_deadline_ms: int = Field(default=12000, ge=500, le=120000)
    max_in_flight_llm: int = Field(default=8, ge=1, le=64)
    max_in_flight_tools: int = Field(default=8, ge=1, le=64)
    tpm_compact_threshold_pct: float = Field(
        default=0.25,
        ge=0.05,
        le=0.50,
        description="Activate Quota-Conserving Compact Mode when remaining TPM <= threshold",
    )
    compact_pieces_k: int = Field(default=2, ge=1, le=4)
    compact_max_tokens: int = Field(default=200, ge=50, le=1024)
    daily_reset_hour_utc: int = Field(
        default=15,
        ge=0,
        le=23,
        description="Daily quota reset hour (15:00 UTC = 3:00 PM UTC)",
    )


class RAGConfig(BaseModel):
    """Corpus, chunking, and retrieval configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    bm25_k1: float = Field(default=1.5, ge=0.1, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=500)
    default_setting: RetrievalSettingType = Field(default="S3_450_8to4_rrf_no")
    corpus_manifest_path: str = Field(default="data/corpus/manifest.json")
    corpus_dir: str = Field(default="data/corpus/pdfs")


class ObservabilityConfig(BaseModel):
    """Logging, metric output, and CSV file paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    log_level: str = Field(default="INFO")
    trace_spans_path: str = Field(default="data/traces/stage_spans.jsonl")
    scenario_metrics_path: str = Field(default="data/traces/scenario_metrics.json")
    results_csv_path: str = Field(default="results.csv")


class RelayConfig(BaseModel):
    """Root configuration model for The Relay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    standin: StandInConfig = Field(default_factory=StandInConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @classmethod
    def load_from_yaml(
        cls,
        yaml_path: Optional[Path] = None,
        env_file_path: Optional[Path] = None,
    ) -> "RelayConfig":
        """Load configuration from YAML file and merge with environment variables."""
        config_dict: Dict[str, Any] = {}

        # 1. Load from YAML if available
        target_yaml = yaml_path or Path("config/default.yaml")
        if target_yaml.is_file():
            with open(target_yaml, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_dict = loaded

        # 2. Extract GROQ_API_KEY from environment or .env file
        api_key_str = os.environ.get("GROQ_API_KEY")
        if not api_key_str:
            target_env = env_file_path or Path(".env")
            if target_env.is_file():
                with open(target_env, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GROQ_API_KEY="):
                            _, val = line.split("=", 1)
                            val = val.strip().strip("'\"")
                            if val and val != "gsk_your_groq_api_key_here":
                                api_key_str = val
                            break

        if api_key_str:
            provider_dict = config_dict.setdefault("provider", {})
            provider_dict["api_key"] = SecretStr(api_key_str)

        return cls.model_validate(config_dict)

