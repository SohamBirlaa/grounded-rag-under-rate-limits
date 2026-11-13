"""Unit tests for Phase 1 data contracts, schemas, and exact CSV enforcement."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from relay.contracts.schemas import (
    CSV_MANDATORY_COLUMNS,
    CitationRecord,
    CSVRowRecord,
    LatencyBreakdown,
    RelayRequest,
    RelayResponse,
    map_label_to_retrieval_setting,
    map_retrieval_setting_to_label,
)


def test_relay_request_valid() -> None:
    """Verify valid request creation with defaults."""
    req = RelayRequest(
        request_id="req_steady_001",
        question="What is the dimension of the key vectors in Multi-Head Attention?",
    )
    assert req.request_id == "req_steady_001"
    assert req.scenario == "steady"
    assert req.source == "real"
    assert req.retrieval_setting == "S3_450_8to4_rrf_no"
    assert req.deadline_ms == 12000


def test_relay_request_extra_forbidden() -> None:
    """Verify extra attributes are strictly forbidden."""
    with pytest.raises(PydanticValidationError):
        RelayRequest(
            request_id="req_002",
            question="What is Moshi?",
            unexpected_field="disallowed",  # type: ignore[call-arg]
        )


def test_citation_record_provenance_validation() -> None:
    """Verify CitationRecord validates paper_id and version alignment."""
    # Valid citation
    cit = CitationRecord(
        paper_id="1706.03762v7",
        version="v7",
        section="3.2.2 Multi-Head Attention",
        page=4,
        chunk_id="chunk_1706_004",
        quoted_evidence="Multi-head attention allows the model to jointly attend to information...",
    )
    assert cit.paper_id == "1706.03762v7"
    assert cit.version == "v7"

    # Version mismatch must raise validation error
    with pytest.raises(PydanticValidationError) as exc:
        CitationRecord(
            paper_id="1706.03762v7",
            version="v1",  # Mismatch: paper_id has v7!
            section="3.2.2 Multi-Head Attention",
            page=4,
            chunk_id="chunk_1706_004",
            quoted_evidence="Multi-head attention allows the model to jointly attend to information...",
        )
    assert "Version 'v1' does not match paper_id '1706.03762v7'" in str(exc.value)


def test_csv_row_record_exact_12_columns() -> None:
    """Verify CSVRowRecord serializes to the exact 12 columns in mandatory order."""
    row = CSVRowRecord(
        request_id="req_001",
        scenario="steady",
        source="real",
        start_time="2026-11-10T02:00:00.123Z",
        first_token_ms=284.5,
        total_ms=1150.2,
        status="ok",
        retries=0,
        tokens_in=850,
        tokens_out=145,
        retrieval_setting="450 tok, 8->4 via RRF, re-rank no",
        answer_grade="correct",
    )

    csv_dict = row.to_csv_dict()
    assert tuple(csv_dict.keys()) == CSV_MANDATORY_COLUMNS
    assert len(csv_dict) == 12

    csv_line = row.to_csv_line()
    expected_line = (
        'req_001,steady,real,2026-11-10T02:00:00.123Z,284.5,1150.2,ok,0,850,145,'
        '"450 tok, 8->4 via RRF, re-rank no",correct'
    )
    assert csv_line == expected_line


def test_csv_row_record_empty_first_token_on_timeout() -> None:
    """Verify that when a request times out or is shed before first token, first_token_ms is empty string."""
    row = CSVRowRecord(
        request_id="req_shed_002",
        scenario="burst",
        source="stand-in",
        start_time="2026-11-10T02:05:00.000Z",
        first_token_ms=None,  # Shed before first token
        total_ms=2.1,
        status="rate_limited",
        retries=0,
        tokens_in=0,
        tokens_out=0,
        retrieval_setting="450 tok, 8->4 via RRF, re-rank no",
        answer_grade="refused_wrongly",
    )

    csv_dict = row.to_csv_dict()
    assert csv_dict["first_token_ms"] == ""
    assert csv_dict["tokens_in"] == "0"
    assert csv_dict["tokens_out"] == "0"
    assert csv_dict["status"] == "rate_limited"

    # Verify round-trip parsing from dict
    parsed = CSVRowRecord.from_csv_dict(csv_dict)
    assert parsed.first_token_ms is None
    assert parsed.status == "rate_limited"
    assert parsed.request_id == "req_shed_002"


def test_csv_row_record_rejects_extra_columns() -> None:
    """Verify that results.csv row parser rejects forbidden extra columns (zero added columns rule)."""
    bad_dict = {
        "request_id": "req_003",
        "scenario": "steady",
        "source": "real",
        "start_time": "2026-11-10T02:00:00.000Z",
        "first_token_ms": "250.0",
        "total_ms": "1200.0",
        "status": "ok",
        "retries": "0",
        "tokens_in": "500",
        "tokens_out": "100",
        "retrieval_setting": "256 tok, 3 pieces, re-rank no",
        "answer_grade": "correct",
        "queue_length": "5",  # Forbidden extra column!
    }
    with pytest.raises(ValueError) as exc:
        CSVRowRecord.from_csv_dict(bad_dict)
    assert "Extra forbidden columns detected" in str(exc.value)


def test_csv_row_record_validates_first_token_not_exceeding_total() -> None:
    """Verify that first_token_ms cannot be greater than total_ms."""
    with pytest.raises(PydanticValidationError) as exc:
        CSVRowRecord(
            request_id="req_004",
            scenario="steady",
            source="real",
            start_time="2026-11-10T02:00:00.000Z",
            first_token_ms=1500.0,  # Greater than total_ms!
            total_ms=1000.0,
            status="ok",
            retries=0,
            tokens_in=500,
            tokens_out=50,
            retrieval_setting="256 tok, 3 pieces, re-rank no",
            answer_grade="correct",
        )
    assert "cannot exceed total_ms" in str(exc.value)


def test_csv_row_record_validates_iso8601() -> None:
    """Verify start_time must be valid ISO-8601 with timezone."""
    with pytest.raises(PydanticValidationError) as exc:
        CSVRowRecord(
            request_id="req_005",
            scenario="steady",
            source="real",
            start_time="invalid-date-string",
            first_token_ms=300.0,
            total_ms=1000.0,
            status="ok",
            retries=0,
            tokens_in=500,
            tokens_out=50,
            retrieval_setting="256 tok, 3 pieces, re-rank no",
            answer_grade="correct",
        )
    assert "Invalid ISO-8601" in str(exc.value)


def test_retrieval_label_mapping() -> None:
    """Verify bidirectional mapping between setting enums and Table 3 labels."""
    assert (
        map_retrieval_setting_to_label("S1_256_3_no")
        == "256 tok, 3 pieces, re-rank no"
    )
    assert (
        map_retrieval_setting_to_label("S2_450_5_no")
        == "450 tok, 5 pieces, re-rank no"
    )
    assert (
        map_retrieval_setting_to_label("S3_450_8to4_rrf_no")
        == "450 tok, 8->4 via RRF, re-rank no"
    )
    assert (
        map_retrieval_setting_to_label("S_opt_structure")
        == "450 tok, 5 pieces, structure-breadcrumb, re-rank no"
    )

    assert (
        map_label_to_retrieval_setting("450 tok, 8->4 via RRF, re-rank no")
        == "S3_450_8to4_rrf_no"
    )

    with pytest.raises(ValueError):
        map_label_to_retrieval_setting("unknown label")


def test_csv_row_record_invalid_status_enum() -> None:
    """Verify that invalid status enums (e.g. 'failed', 'cancelled') are strictly rejected."""
    with pytest.raises(PydanticValidationError):
        CSVRowRecord(
            request_id="req_006",
            scenario="steady",
            source="real",
            start_time="2026-11-10T02:00:00.000Z",
            first_token_ms=250.0,
            total_ms=1000.0,
            status="failed",  # Invalid! Must be one of ok, rate_limited, timeout, error
            retries=0,
            tokens_in=500,
            tokens_out=50,
            retrieval_setting="256 tok, 3 pieces, re-rank no",
            answer_grade="correct",
        )


def test_csv_row_record_invalid_answer_grade_enum() -> None:
    """Verify that invalid answer_grade enums are strictly rejected."""
    with pytest.raises(PydanticValidationError):
        CSVRowRecord(
            request_id="req_007",
            scenario="steady",
            source="real",
            start_time="2026-11-10T02:00:00.000Z",
            first_token_ms=250.0,
            total_ms=1000.0,
            status="ok",
            retries=0,
            tokens_in=500,
            tokens_out=50,
            retrieval_setting="256 tok, 3 pieces, re-rank no",
            answer_grade="partially_correct",  # Invalid! Must be correct, wrong, refused_correctly, refused_wrongly
        )


def test_relay_response_full_model() -> None:
    """Verify full RelayResponse serialization with LatencyBreakdown."""
    resp = RelayResponse(
        request_id="req_008",
        status="ok",
        is_refusal=False,
        answer_text="Multi-Head Attention projects queries, keys, and values h times [arXiv:1706.03762v7, Sec 3.2.2, p. 4].",
        citations=[
            CitationRecord(
                paper_id="1706.03762v7",
                version="v7",
                section="3.2.2 Multi-Head Attention",
                page=4,
                chunk_id="chunk_1706_004",
                quoted_evidence="Multi-head attention allows the model to jointly attend to information...",
            )
        ],
        tokens_in=850,
        tokens_out=120,
        retries=0,
        first_token_ms=280.0,
        total_ms=1100.0,
        latency_breakdown=LatencyBreakdown(
            waiting_ms=10.0,
            working_ms=1090.0,
            retrying_ms=0.0,
            stage1_query_ms=2.0,
            stage2_retrieval_ms=8.0,
            stage3_generate_ms=1050.0,
            stage4_verify_ms=30.0,
            pre_first_token_ms=280.0,
            post_first_token_ms=820.0,
        ),
        answer_grade="correct",
    )
    assert resp.status == "ok"
    assert len(resp.citations) == 1
    assert resp.citations[0].page == 4
    assert resp.latency_breakdown.stage4_verify_ms == 30.0

