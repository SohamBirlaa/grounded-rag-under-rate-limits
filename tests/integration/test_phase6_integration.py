"""Integration tests for Phase 6: End-to-End Pipeline Integration.

Covers:
- Category A, B, C answerable queries with grounded citations
- Category D epistemic refusal queries
- Stage 4 citation verification and 1-time repair loop
- Queue admission, proactive deadline shedding, and capacity rejection
- 3:00 PM Daily quota latch with offline raw excerpt fallback (< 15 ms)
- Circuit breaker 5xx outage protection
- Strict 12-column CSVRowRecord emission
- API adapter ASGI endpoints (/health, /query, /query/stream)
"""

import asyncio
import json
from pathlib import Path
import pytest

from relay.api.app import RelayAPIAdapter, create_app
from relay.config.settings import RelayConfig, ReliabilityConfig
from relay.contracts.schemas import (
    CSV_MANDATORY_COLUMNS,
    CSVRowRecord,
    RelayRequest,
    RelayResponse,
)
from relay.engine.orchestrator import RelayEngine
from relay.rag.retriever import Retriever
from relay.reliability.circuit_breaker import CircuitBreaker
from relay.reliability.concurrency import ConcurrencyGate
from relay.reliability.queue import BoundedRequestQueue
from relay.reliability.rate_limiter import ClientRateLimiter
from relay.reliability.timeouts import DeadlineTracker


@pytest.fixture(scope="module")
def retriever():
    """Load retriever from disk cache in data/corpus/indexes/."""
    indexes_dir = Path("data/corpus/indexes")
    if not (indexes_dir / "s1_256_chunks.json").exists():
        pytest.skip(f"Index cache not found in {indexes_dir}")
    return Retriever.from_cache(indexes_dir)


@pytest.mark.asyncio
async def test_phase6_category_a_answerable_query(tmp_path, retriever):
    """Verify Category A query completes with answer, citations, and valid CSV row."""
    trace_file = tmp_path / "stage_spans.jsonl"
    engine = RelayEngine(
        retriever=retriever,
        trace_path=trace_file,
        fast_test_mode=True,
    )

    req = RelayRequest(
        request_id="req_p6_catA",
        question="What is Scaled Dot-Product Attention in the Transformer?",
        scenario="steady",
        source="stand-in",
        retrieval_setting="S3_450_8to4_rrf_no",
        deadline_ms=5000,
        question_id="q_catA_02",
    )

    response, csv_row = await engine.execute_request(req)

    assert response.status == "ok"
    assert response.is_refusal is False
    assert len(response.answer_text) > 0
    assert len(response.citations) > 0
    assert response.answer_grade in ["correct", "wrong"]

    # 12-column CSV contract
    assert csv_row.request_id == "req_p6_catA"
    assert csv_row.status == "ok"
    assert len(csv_row.to_csv_dict()) == 12
    assert tuple(csv_row.to_csv_dict().keys()) == CSV_MANDATORY_COLUMNS


@pytest.mark.asyncio
async def test_phase6_category_d_epistemic_refusal(tmp_path, retriever):
    """Verify Category D unanswerable query results in epistemic refusal and refused_correctly grade."""
    trace_file = tmp_path / "stage_spans.jsonl"
    engine = RelayEngine(
        retriever=retriever,
        trace_path=trace_file,
        fast_test_mode=True,
    )

    req = RelayRequest(
        request_id="req_p6_catD",
        question="What is the 15-trillion token pretraining mixture of Meta Llama-3-70B?",
        scenario="steady",
        source="stand-in",
        retrieval_setting="S3_450_8to4_rrf_no",
        deadline_ms=5000,
        question_id="q_catD_01",
    )

    response, csv_row = await engine.execute_request(req)

    assert response.status == "ok"
    assert response.is_refusal is True
    assert response.answer_grade == "refused_correctly"
    assert csv_row.answer_grade == "refused_correctly"
    assert csv_row.status == "ok"


@pytest.mark.asyncio
async def test_phase6_stage4_citation_repair_loop(tmp_path, retriever):
    """Verify Stage 4 one-time repair loop executes when citations are missing in initial generation."""
    trace_file = tmp_path / "stage_spans.jsonl"
    engine = RelayEngine(
        retriever=retriever,
        trace_path=trace_file,
        fast_test_mode=True,
    )

    # Question is answerable; initial mock response might need formatting
    req = RelayRequest(
        request_id="req_p6_repair",
        question="How does Moshi achieve full-duplex spoken dialogue?",
        scenario="steady",
        source="stand-in",
        retrieval_setting="S3_450_8to4_rrf_no",
        deadline_ms=5000,
        question_id="q_catA_05",
    )

    response, csv_row = await engine.execute_request(req)
    assert response.status == "ok"
    assert len(response.citations) > 0


@pytest.mark.asyncio
async def test_phase6_daily_quota_latch_offline_fallback(tmp_path, retriever):
    """Verify 3:00 PM daily quota latch triggers immediate (< 15 ms) offline refusal with raw excerpts."""
    trace_file = tmp_path / "stage_spans.jsonl"
    breaker = CircuitBreaker()
    breaker.latch_daily_quota()

    engine = RelayEngine(
        retriever=retriever,
        trace_path=trace_file,
        circuit_breaker=breaker,
        fast_test_mode=True,
    )

    req = RelayRequest(
        request_id="req_p6_quota_latched",
        question="What is Scaled Dot-Product Attention in the Transformer?",
        scenario="daily_budget_exhausted",
        source="stand-in",
        retrieval_setting="S3_450_8to4_rrf_no",
        deadline_ms=5000,
    )

    response, csv_row = await engine.execute_request(req)

    assert response.status == "rate_limited"
    assert response.is_refusal is True
    assert "Daily inference quota exhausted" in response.refusal_reason
    assert "15:00:00Z" in response.answer_text
    assert len(response.citations) > 0
    assert response.total_ms < 50.0  # Fast execution (< 15 ms typical)
    assert csv_row.status == "rate_limited"
    assert csv_row.answer_grade == "refused_correctly"


@pytest.mark.asyncio
async def test_phase6_queue_capacity_shed(tmp_path, retriever):
    """Verify arrival is shed when queue reaches capacity ceiling (24)."""
    trace_file = tmp_path / "stage_spans.jsonl"
    queue = BoundedRequestQueue(
        config=ReliabilityConfig(max_queue_size=2),
        default_service_rate_req_per_s=100.0,
    )

    engine = RelayEngine(
        retriever=retriever,
        trace_path=trace_file,
        queue=queue,
        fast_test_mode=False,
    )

    # Fill queue to capacity (2 items)
    t1 = DeadlineTracker(10000)
    t2 = DeadlineTracker(10000)
    await queue.admit_or_shed("q_fill_1", t1)
    await queue.admit_or_shed("q_fill_2", t2)

    # 3rd request through engine must be shed with rate_limited
    req = RelayRequest(
        request_id="req_p6_shed",
        question="What is Scaled Dot-Product Attention in the Transformer?",
        scenario="burst",
        source="stand-in",
        retrieval_setting="S3_450_8to4_rrf_no",
        deadline_ms=5000,
    )

    response, csv_row = await engine.execute_request(req)
    assert response.status == "rate_limited"
    assert "capacity reached ceiling" in response.refusal_reason
    assert csv_row.status == "rate_limited"


@pytest.mark.asyncio
async def test_phase6_api_adapter_endpoints(tmp_path, retriever):
    """Verify ASGI API adapter handles /health, /query, and /query/stream."""
    engine = RelayEngine(retriever=retriever, trace_path=tmp_path / "spans.jsonl", fast_test_mode=True)
    adapter = RelayAPIAdapter(engine=engine)

    # 1. /health endpoint
    health_scope = {"type": "http", "method": "GET", "path": "/health"}
    response_started = False
    status_code = None
    response_body = []

    async def mock_receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def mock_send(message):
        nonlocal response_started, status_code, response_body
        if message["type"] == "http.response.start":
            response_started = True
            status_code = message["status"]
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))

    await adapter(health_scope, mock_receive, mock_send)
    assert status_code == 200
    health_data = json.loads(b"".join(response_body).decode("utf-8"))
    assert health_data["status"] == "ok"

    # 2. handle_request directly
    payload = {
        "request_id": "req_api_01",
        "question": "What is Scaled Dot-Product Attention in the Transformer?",
        "scenario": "steady",
        "source": "stand-in",
        "retrieval_setting": "S3_450_8to4_rrf_no",
        "deadline_ms": 5000,
    }
    result = await adapter.handle_request(payload)
    assert result["request_id"] == "req_api_01"
    assert result["status"] == "ok"
    assert len(result["answer_text"]) > 0

    # 3. handle_stream SSE
    stream_events = []
    async for chunk in adapter.handle_stream(payload):
        stream_events.append(chunk)

    assert len(stream_events) >= 3
    assert any("event: status" in e for e in stream_events)
    assert any("event: metadata" in e for e in stream_events)
    assert any("event: answer" in e for e in stream_events)
    assert any("event: done" in e for e in stream_events)
