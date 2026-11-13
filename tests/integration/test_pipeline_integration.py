"""Integration tests for the complete Relay pipeline across RAG, Providers, and Reliability."""

import asyncio
import json
from pathlib import Path
import pytest
import httpx

from relay.config.settings import RelayConfig
from relay.contracts.schemas import (
    CSV_MANDATORY_COLUMNS,
    CSVRowRecord,
    RelayRequest,
    RelayResponse,
)
from relay.engine.orchestrator import RelayEngine
from relay.providers.groq import GroqProviderAdapter
from relay.rag.retriever import Retriever


@pytest.fixture(scope="module")
def retriever():
    """Load retriever from disk cache in data/corpus/indexes/."""
    indexes_dir = Path("data/corpus/indexes")
    if not (indexes_dir / "s1_256_chunks.json").exists():
        pytest.skip(f"Index cache not found in {indexes_dir}")
    return Retriever.from_cache(indexes_dir)


def test_end_to_end_pipeline_with_standin(tmp_path, retriever):
    """Verify end-to-end request processing with stand-in emits valid response and 12-column CSV row."""
    async def _run():
        trace_file = tmp_path / "stage_spans.jsonl"
        engine = RelayEngine(
            retriever=retriever,
            trace_path=trace_file,
            fast_test_mode=True,
        )

        request = RelayRequest(
            request_id="req_integration_01",
            question="What is Scaled Dot-Product Attention in the Transformer?",
            scenario="steady",
            source="stand-in",
            retrieval_setting="S3_450_8to4_rrf_no",
            deadline_ms=5000,
            question_id="q_catA_02",
        )

        response, csv_row = await engine.execute_request(request)

        # 1. Verify RelayResponse structure
        assert isinstance(response, RelayResponse)
        assert response.request_id == "req_integration_01"
        assert response.status == "ok"
        assert response.first_token_ms is not None
        assert response.first_token_ms <= response.total_ms
        assert len(response.answer_text) > 0
        assert len(response.citations) > 0

        # 2. Verify CSVRowRecord exact 12-column contract
        assert isinstance(csv_row, CSVRowRecord)
        assert csv_row.request_id == "req_integration_01"
        assert csv_row.scenario == "steady"
        assert csv_row.source == "stand-in"
        assert csv_row.status == "ok"
        assert csv_row.retries == 0
        assert csv_row.retrieval_setting == "450 tok, 8->4 via RRF, re-rank no"

        # Strict 12-column CSV line formatting
        csv_line = csv_row.to_csv_line()
        import csv
        import io
        csv_reader = csv.reader(io.StringIO(csv_line))
        csv_parts = next(csv_reader)
        assert len(csv_parts) == 12
        assert len(CSV_MANDATORY_COLUMNS) == 12

        # Round-trip dictionary validation
        row_dict = csv_row.to_csv_dict()
        assert tuple(row_dict.keys()) == CSV_MANDATORY_COLUMNS

        # 3. Verify diagnostic trace span written
        assert trace_file.is_file()
        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.loads(f.readline())
            assert trace_data["request_id"] == "req_integration_01"
            assert trace_data["source"] == "stand-in"
            assert trace_data["stage2_retrieval_ms"] >= 0.0
            assert trace_data["stage3_generate_ms"] >= 0.0

    asyncio.run(_run())


def test_end_to_end_epistemic_refusal_on_cat_d(tmp_path, retriever):
    """Verify Cat D corpus-absent question results in epistemic refusal and refused_correctly grade."""
    async def _run():
        trace_file = tmp_path / "stage_spans.jsonl"
        engine = RelayEngine(
            retriever=retriever,
            trace_path=trace_file,
            fast_test_mode=True,
        )

        request = RelayRequest(
            request_id="req_integration_catD",
            question="What is the 15-trillion token pretraining mixture of Meta Llama-3-70B?",
            scenario="steady",
            source="stand-in",
            retrieval_setting="S3_450_8to4_rrf_no",
            deadline_ms=5000,
            question_id="q_catD_01",
        )

        response, csv_row = await engine.execute_request(request)

        assert response.status == "ok"
        assert response.is_refusal is True
        assert "cannot answer" in response.answer_text.lower()
        assert csv_row.answer_grade == "refused_correctly"

    asyncio.run(_run())


def test_strict_provider_isolation_no_silent_fallback(tmp_path, retriever):
    """Verify that a failure on source='real' NEVER silently falls back to stand-in."""
    async def _run():
        trace_file = tmp_path / "stage_spans.jsonl"

        # Mock Groq provider that simulates a 401 Unauthorized rejection
        async def mock_handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

        transport = httpx.MockTransport(mock_handler)
        mock_client = httpx.AsyncClient(transport=transport)

        mock_groq = GroqProviderAdapter(
            api_key=None,  # No API key -> raises ProviderAuthenticationError
            http_client=mock_client,
        )

        engine = RelayEngine(
            retriever=retriever,
            trace_path=trace_file,
            fast_test_mode=True,
            provider_override=mock_groq,
        )

        request = RelayRequest(
            request_id="req_real_failure",
            question="What is the Transformer architecture?",
            scenario="steady",
            source="real",  # Explicitly requesting real provider
            retrieval_setting="S3_450_8to4_rrf_no",
            deadline_ms=5000,
        )

        response, csv_row = await engine.execute_request(request)

        # Must fail cleanly as real, NEVER fallback to stand-in!
        assert response.status == "error"
        assert csv_row.source == "real"
        assert csv_row.status == "error"
        assert csv_row.first_token_ms is None

        await mock_client.aclose()

    asyncio.run(_run())
