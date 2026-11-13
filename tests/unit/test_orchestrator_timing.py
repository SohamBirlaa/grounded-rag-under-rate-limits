"""Unit tests for request-level versus provider-level timing semantics in RelayEngine.

Verifies:
- Request-level first_token_ms shares the exact same origin (t0) as total_ms.
- Retrieval duration (Stage 2) is included in request-level first_token_ms.
- Provider-only TTFT is preserved separately in stage span details.
- first_token_ms is strictly None when request fails or times out before first token.
- Exact 12-column CSV line formatting with proper quoting.
"""

import asyncio
import csv
import io
import json
from pathlib import Path
import time
from typing import AsyncIterator
import pytest

from relay.contracts.schemas import (
    CSV_MANDATORY_COLUMNS,
    CSVRowRecord,
    RelayRequest,
    RelayResponse,
)
from relay.engine.orchestrator import RelayEngine
from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
    ProviderQuotaSnapshot,
    ProviderStreamChunk,
)


class MockDelayedProvider:
    """Mock provider with controllable artificial TTFT and chunk generation delay."""

    def __init__(self, ttft_s: float = 0.05, total_chunks: int = 3) -> None:
        self.ttft_s = ttft_s
        self.total_chunks = total_chunks

    def get_quota_snapshot(self) -> ProviderQuotaSnapshot:
        return ProviderQuotaSnapshot()

    async def check_health(self) -> bool:
        return True

    async def stream_completion(
        self, request: ProviderCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        if self.ttft_s > 0:
            await asyncio.sleep(self.ttft_s)

        arrival_s = time.perf_counter()
        for i in range(1, self.total_chunks + 1):
            yield ProviderStreamChunk(
                text_delta=f"token_{i} ",
                is_first_token=(i == 1),
                token_index=i,
                arrival_monotonic_s=arrival_s if i == 1 else time.perf_counter(),
                prompt_tokens=15,
                completion_tokens=self.total_chunks,
            )


def test_orchestrator_timing_request_vs_provider_ttft(tmp_path):
    """Verify request-level first_token_ms is measured from request arrival t0 and includes Stage 1/2."""
    async def _run():
        trace_file = tmp_path / "stage_spans.jsonl"
        mock_provider = MockDelayedProvider(ttft_s=0.04)

        engine = RelayEngine(
            retriever=None,  # No retriever: fast Stage 2
            trace_path=trace_file,
            fast_test_mode=True,
            provider_override=mock_provider,
        )

        request = RelayRequest(
            request_id="req_timing_01",
            question="What is the Transformer architecture in Attention is all you need?",
            scenario="steady",
            source="stand-in",
            retrieval_setting="S3_450_8to4_rrf_no",
            deadline_ms=5000,
        )

        response, csv_row = await engine.execute_request(request)

        # 1. Verification of CSVRowRecord
        assert csv_row.first_token_ms is not None
        assert csv_row.total_ms > 0
        assert csv_row.first_token_ms <= csv_row.total_ms

        # 2. Strict 12-column CSV formatting
        csv_line = csv_row.to_csv_line()
        csv_parts = next(csv.reader(io.StringIO(csv_line)))
        assert len(csv_parts) == 12
        assert len(CSV_MANDATORY_COLUMNS) == 12
        assert csv_parts[0] == "req_timing_01"
        assert csv_parts[1] == "steady"
        assert csv_parts[2] == "stand-in"
        assert csv_parts[10] == "450 tok, 8->4 via RRF, re-rank no"

        # 3. Verification of StageSpanRecord
        assert trace_file.is_file()
        with open(trace_file, "r", encoding="utf-8") as f:
            span_json = json.loads(f.readline())
            assert span_json["request_id"] == "req_timing_01"
            assert span_json["first_token_ms"] == csv_row.first_token_ms
            assert span_json["total_ms"] == csv_row.total_ms
            # Provider-only TTFT is preserved in details
            provider_ttft = span_json["details"]["provider_ttft_ms"]
            assert provider_ttft is not None
            # Request-level TTFT must be >= provider-only TTFT because it includes stage 1 and stage 2 overhead
            assert span_json["first_token_ms"] >= provider_ttft

    asyncio.run(_run())


def test_orchestrator_timing_empty_first_token_on_timeout(tmp_path):
    """Verify first_token_ms is strictly None when request times out before receiving first token."""
    async def _run():
        trace_file = tmp_path / "stage_spans.jsonl"
        # Mock provider that sleeps longer than deadline
        slow_provider = MockDelayedProvider(ttft_s=0.8)

        engine = RelayEngine(
            retriever=None,
            trace_path=trace_file,
            fast_test_mode=True,
            provider_override=slow_provider,
        )

        # Request with 500ms deadline (contract minimum)
        request = RelayRequest(
            request_id="req_timeout_before_first_token",
            question="What is the Transformer architecture?",
            scenario="steady",
            source="stand-in",
            retrieval_setting="S3_450_8to4_rrf_no",
            deadline_ms=500,
        )

        response, csv_row = await engine.execute_request(request)

        assert response.status == "timeout"
        assert response.first_token_ms is None
        assert csv_row.status == "timeout"
        assert csv_row.first_token_ms is None

        # Formatted CSV line must have empty string for first_token_ms (column index 4)
        csv_line = csv_row.to_csv_line()
        csv_parts = next(csv.reader(io.StringIO(csv_line)))
        assert len(csv_parts) == 12
        assert csv_parts[4] == ""
        assert float(csv_parts[5]) > 0.0  # total_ms must be populated

    asyncio.run(_run())
