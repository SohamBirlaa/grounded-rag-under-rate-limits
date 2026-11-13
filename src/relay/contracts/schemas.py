"""Data models and API contracts for The Relay.

Implements Pydantic v2 immutable schemas with strict extra-attribute forbidding.
Locks the exact 12-column results.csv schema mandated by the assignment brief.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Core Literal Types ---

ScenarioType = Literal[
    "discovery",
    "steady",
    "burst",
    "sustained_overload",
    "rpm_halved",
    "daily_budget_exhausted",
    "retrieval_eval",
]

ProviderSourceType = Literal["real", "stand-in"]

RetrievalSettingType = Literal[
    "S1_256_3_no",
    "S2_450_5_no",
    "S3_450_8to4_rrf_no",
    "S_opt_structure",
]

Table3RetrievalSettingLabel = Literal[
    "256 tok, 3 pieces, re-rank no",
    "450 tok, 5 pieces, re-rank no",
    "450 tok, 8->4 via RRF, re-rank no",
    "450 tok, 5 pieces, structure-breadcrumb, re-rank no",
]

PaperIdType = Literal["1706.03762v7", "2410.00037v1", "2507.07955v2"]

RequestStatusType = Literal["ok", "rate_limited", "timeout", "error"]

AnswerGradeType = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]

# Mapping dictionaries
SETTING_TO_LABEL_MAP: Dict[RetrievalSettingType, str] = {
    "S1_256_3_no": "256 tok, 3 pieces, re-rank no",
    "S2_450_5_no": "450 tok, 5 pieces, re-rank no",
    "S3_450_8to4_rrf_no": "450 tok, 8->4 via RRF, re-rank no",
    "S_opt_structure": "450 tok, 5 pieces, structure-breadcrumb, re-rank no",
}

LABEL_TO_SETTING_MAP: Dict[str, RetrievalSettingType] = {
    v: k for k, v in SETTING_TO_LABEL_MAP.items()
}


def map_retrieval_setting_to_label(setting: RetrievalSettingType) -> str:
    """Map internal retrieval setting enum to canonical Table 3 label."""
    if setting in SETTING_TO_LABEL_MAP:
        return SETTING_TO_LABEL_MAP[setting]
    raise ValueError(f"Unknown retrieval setting: {setting}")


def map_label_to_retrieval_setting(label: str) -> RetrievalSettingType:
    """Map canonical Table 3 label to internal retrieval setting enum."""
    if label in LABEL_TO_SETTING_MAP:
        return LABEL_TO_SETTING_MAP[label]
    raise ValueError(f"Unknown Table 3 retrieval label: {label}")


# --- 1. Request Schema ---


class RelayRequest(BaseModel):
    """Client request model representing a research query entering The Relay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(
        ...,
        min_length=4,
        max_length=64,
        description="Unique deterministic or UUID request identifier",
    )
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Engineer's research question",
    )
    scenario: ScenarioType = Field(
        default="steady",
        description="Load or evaluation scenario name",
    )
    source: ProviderSourceType = Field(
        default="real",
        description="Target model backend: 'real' (Groq) or 'stand-in'",
    )
    retrieval_setting: RetrievalSettingType = Field(
        default="S3_450_8to4_rrf_no",
        description="Retrieval configuration identifier recorded in results.csv",
    )
    deadline_ms: int = Field(
        default=12000,
        ge=500,
        le=60000,
        description="Client patience deadline in milliseconds",
    )
    question_id: Optional[str] = Field(
        default=None,
        description="Optional reference ID in evaluation dataset (e.g. 'q_catA_01')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional benchmark trace tags",
    )


# --- 2. Citation Schema ---


class CitationRecord(BaseModel):
    """Structured citation marker pointing to exact arXiv paper provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: PaperIdType = Field(
        ...,
        description="Exact arXiv ID with mandatory version suffix",
    )
    version: Literal["v7", "v1", "v2"] = Field(
        ...,
        description="Verified paper version",
    )
    section: str = Field(
        ...,
        min_length=1,
        max_length=160,
        description="Section number and heading in paper",
    )
    page: int = Field(
        ...,
        ge=1,
        le=200,
        description="1-indexed PDF page number",
    )
    chunk_id: str = Field(
        ...,
        min_length=4,
        description="Deterministic chunk ID in corpus index",
    )
    quoted_evidence: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Supporting verbatim span extracted from chunk",
    )

    @field_validator("version")
    @classmethod
    def validate_version_matches_paper(cls, v: str, info: Any) -> str:
        paper_id = info.data.get("paper_id")
        if paper_id and not paper_id.endswith(v):
            raise ValueError(f"Version '{v}' does not match paper_id '{paper_id}'")
        return v


# --- 3. Latency Breakdown Schema (Heading 4 Diagnostics) ---


class LatencyBreakdown(BaseModel):
    """Monotonic latency breakdown for diagnostic auditing in Heading 4."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    waiting_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent waiting in queue + admission pacing",
    )
    working_ms: float = Field(
        ...,
        ge=0.0,
        description="Active pipeline execution time across Stages 1-4",
    )
    retrying_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Failed attempt duration + retry backoff delay",
    )
    stage1_query_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Stage 1 query analysis duration",
    )
    stage2_retrieval_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Stage 2 tool execution duration",
    )
    stage3_generate_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Stage 3 streaming generation duration",
    )
    stage4_verify_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Stage 4 deterministic citation verification duration",
    )
    pre_first_token_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Delay before first token chunk arrives",
    )
    post_first_token_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Streaming + verification delay after first token",
    )


# --- 4. Answer Schema (`RelayResponse`) ---


class RelayResponse(BaseModel):
    """Complete response returned by the RelayEngine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    status: RequestStatusType
    is_refusal: bool = Field(
        ...,
        description="True if assistant abstained due to missing evidence or quota",
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Explanation when is_refusal=True or status!='ok'",
    )
    answer_text: str = Field(
        ...,
        description="Final cited answer or user-facing refusal/degradation notice",
    )
    citations: List[CitationRecord] = Field(default_factory=list)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    first_token_ms: Optional[float] = Field(default=None, ge=0.0)
    total_ms: float = Field(..., ge=0.0)
    latency_breakdown: LatencyBreakdown
    answer_grade: Optional[AnswerGradeType] = Field(
        default=None,
        description="Graded quality outcome recorded in results.csv",
    )


# --- 5. Mandatory `results.csv` Row Contract ---

CSV_MANDATORY_COLUMNS: tuple[str, ...] = (
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
)


class CSVRowRecord(BaseModel):
    """Strict schema enforcing the exact 12-column results.csv contract.

    Mandated on Page 4 of the assignment brief:
    request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(..., min_length=1, max_length=64)
    scenario: ScenarioType
    source: ProviderSourceType
    start_time: str = Field(
        ...,
        description="ISO-8601 UTC timestamp string (e.g. 2026-11-10T02:30:00.123Z)",
    )
    first_token_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time to first token in ms (None if shed/timed out before 1st token)",
    )
    total_ms: float = Field(
        ...,
        ge=0.0,
        description="Total elapsed wall time from arrival to completion in ms",
    )
    status: RequestStatusType
    retries: int = Field(..., ge=0)
    tokens_in: int = Field(..., ge=0)
    tokens_out: int = Field(..., ge=0)
    retrieval_setting: str = Field(
        ...,
        description="Active Table 3 retrieval setting label",
    )
    answer_grade: AnswerGradeType

    @field_validator("start_time")
    @classmethod
    def validate_iso8601(cls, v: str) -> str:
        """Verify that start_time is valid ISO-8601."""
        try:
            # Normalize trailing Z for Python fromisoformat
            norm_str = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(norm_str)
            if dt.tzinfo is None:
                raise ValueError("start_time must include timezone offset (e.g. 'Z')")
        except Exception as e:
            raise ValueError(f"Invalid ISO-8601 start_time '{v}': {e}") from e
        return v

    @model_validator(mode="after")
    def validate_first_token_against_total(self) -> "CSVRowRecord":
        """Verify first_token_ms does not exceed total_ms when present."""
        if self.first_token_ms is not None and self.total_ms is not None:
            if self.first_token_ms > self.total_ms + 1.0:  # Allow 1ms floating point margin
                raise ValueError(
                    f"first_token_ms ({self.first_token_ms:.1f}) cannot exceed total_ms ({self.total_ms:.1f})"
                )
        return self

    def to_csv_dict(self) -> Dict[str, str]:
        """Convert record to a dictionary of string values matching results.csv."""
        return {
            "request_id": self.request_id,
            "scenario": self.scenario,
            "source": self.source,
            "start_time": self.start_time,
            "first_token_ms": f"{self.first_token_ms:.1f}" if self.first_token_ms is not None else "",
            "total_ms": f"{self.total_ms:.1f}",
            "status": self.status,
            "retries": str(self.retries),
            "tokens_in": str(self.tokens_in),
            "tokens_out": str(self.tokens_out),
            "retrieval_setting": self.retrieval_setting,
            "answer_grade": self.answer_grade,
        }

    def to_csv_line(self) -> str:
        """Format as a single comma-separated CSV line in the exact 12-column order."""
        import csv
        import io

        row_dict = self.to_csv_dict()
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="")
        writer.writerow([row_dict[col] for col in CSV_MANDATORY_COLUMNS])
        return output.getvalue()

    @classmethod
    def from_csv_dict(cls, row: Dict[str, str]) -> "CSVRowRecord":
        """Construct and validate CSVRowRecord from raw string dict (e.g. from csv.DictReader)."""
        extra_keys = set(row.keys()) - set(CSV_MANDATORY_COLUMNS)
        if extra_keys:
            raise ValueError(f"Extra forbidden columns detected in results.csv: {extra_keys}")

        first_token_raw = row.get("first_token_ms", "").strip()
        first_token_val: Optional[float] = float(first_token_raw) if first_token_raw else None

        return cls(
            request_id=row["request_id"].strip(),
            scenario=row["scenario"].strip(),  # type: ignore[arg-type]
            source=row["source"].strip(),  # type: ignore[arg-type]
            start_time=row["start_time"].strip(),
            first_token_ms=first_token_val,
            total_ms=float(row["total_ms"].strip()),
            status=row["status"].strip(),  # type: ignore[arg-type]
            retries=int(row["retries"].strip()),
            tokens_in=int(row["tokens_in"].strip()),
            tokens_out=int(row["tokens_out"].strip()),
            retrieval_setting=row["retrieval_setting"].strip(),
            answer_grade=row["answer_grade"].strip(),  # type: ignore[arg-type]
        )


# --- 6. Supplementary Telemetry Records ---


class StageSpanRecord(BaseModel):
    """Schema for individual request spans recorded in data/traces/stage_spans.jsonl."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    scenario: str
    source: str
    start_time_iso: str
    waiting_ms: float
    working_ms: float
    retrying_ms: float
    stage1_query_ms: float
    stage2_retrieval_ms: float
    stage3_generate_ms: float
    stage4_verify_ms: float
    first_token_ms: Optional[float] = None
    total_ms: float
    pre_ttft_delay_ms: Optional[float] = None
    post_ttft_delay_ms: Optional[float] = None
    slow_request_flag: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class QueueTransitionEvent(BaseModel):
    """Single queue depth transition logged by the atomic queue gauge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_ms: float
    event: Literal["enqueue", "dequeue", "shed", "timeout_evict"]
    queue_depth: int
    request_id: str


class ScenarioMetricsRecord(BaseModel):
    """Supplementary telemetry record written to data/traces/scenario_metrics.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario: str
    source: str
    duration_s: float
    total_requests: int
    completed_requests: int
    direct_peak_queue_length: int
    reconstructed_peak_queue_length: Optional[int] = None
    queue_estimation_error: Optional[int] = None
    queue_reconstruction_valid: bool = True
    transitions_count: int = 0
