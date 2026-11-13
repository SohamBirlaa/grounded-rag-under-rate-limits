"""Unit tests for the GroqProviderAdapter using mock HTTP transports.

Guarantees 100% offline verification with ZERO live network or API key calls.
Verifies response lifecycle, stream cleanup, client ownership, and error normalization.
"""

import asyncio
import json
from typing import Optional
import pytest
import httpx
from pydantic import SecretStr

from relay.contracts.errors import (
    DailyBudgetExhaustedError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from relay.providers.base import ProviderCompletionRequest
from relay.providers.groq import GroqProviderAdapter


@pytest.fixture
def sample_request():
    return ProviderCompletionRequest(
        messages=[{"role": "user", "content": "What is the Transformer?"}],
        model_id="llama-3.1-8b-instant",
        request_id="req_test_01",
    )


def test_groq_adapter_missing_api_key(sample_request):
    """Verify adapter raises ProviderAuthenticationError if no API key is provided."""
    async def _run():
        adapter = GroqProviderAdapter(api_key=None)
        with pytest.raises(ProviderAuthenticationError, match="GROQ_API_KEY is not configured"):
            async for _ in adapter.stream_completion(sample_request):
                pass
    asyncio.run(_run())


def test_groq_adapter_successful_sse_stream(sample_request):
    """Verify parsing of standard OpenAI/Groq SSE chunks and usage metadata."""
    async def _run():
        sse_lines = [
            'data: {"id":"chat_1","choices":[{"delta":{"content":"The "}}]}\n\n',
            'data: {"id":"chat_1","choices":[{"delta":{"content":"Transformer "}}]}\n\n',
            'data: {"id":"chat_1","choices":[{"delta":{"content":"model."}}]}\n\n',
            'data: {"id":"chat_1","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":3}}\n\n',
            "data: [DONE]\n\n",
        ]

        captured_responses = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("authorization") == "Bearer gsk_dummy_test_key_12345"
            headers = {
                "Content-Type": "text/event-stream",
                "x-ratelimit-remaining-requests": "14350",
                "x-ratelimit-remaining-tokens": "5900",
            }
            resp = httpx.Response(200, headers=headers, text="".join(sse_lines))
            captured_responses.append(resp)
            return resp

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key_12345"),
                http_client=client,
            )

            chunks = []
            async for chunk in adapter.stream_completion(sample_request):
                chunks.append(chunk)

            assert len(chunks) == 4  # 3 content chunks + 1 final usage chunk
            assert chunks[0].is_first_token is True
            assert chunks[0].text_delta == "The "
            assert chunks[1].is_first_token is False
            assert chunks[1].text_delta == "Transformer "
            assert chunks[2].text_delta == "model."

            # Verify final usage extraction
            final_chunk = chunks[-1]
            assert final_chunk.prompt_tokens == 25
            assert final_chunk.completion_tokens == 3

            # Verify quota snapshot was updated from headers
            snapshot = adapter.get_quota_snapshot()
            assert snapshot.rpd_remaining == 14350
            assert snapshot.tpm_remaining == 5900

            # Verify response stream cleanup (Issue B)
            assert len(captured_responses) == 1
            assert captured_responses[0].is_closed is True
            # External client must remain OPEN and reusable
            assert client.is_closed is False

    asyncio.run(_run())


def test_groq_adapter_streaming_response_closed_on_early_termination(sample_request):
    """Verify streamed response is closed when consumer breaks out early."""
    async def _run():
        sse_lines = [
            'data: {"id":"chat_1","choices":[{"delta":{"content":"Chunk 1"}}]}\n\n',
            'data: {"id":"chat_1","choices":[{"delta":{"content":"Chunk 2"}}]}\n\n',
            'data: {"id":"chat_1","choices":[{"delta":{"content":"Chunk 3"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        captured_responses = []

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            resp = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text="".join(sse_lines))
            captured_responses.append(resp)
            return resp

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )

            gen = adapter.stream_completion(sample_request)
            # Read only first chunk and stop
            chunk1 = await gen.__anext__()
            assert chunk1.text_delta == "Chunk 1"
            await gen.aclose()

            assert len(captured_responses) == 1
            assert captured_responses[0].is_closed is True
            assert client.is_closed is False

    asyncio.run(_run())


def test_groq_adapter_streaming_response_closed_on_read_error(sample_request):
    """Verify streamed response is closed when read exception occurs mid-stream."""
    async def _run():
        captured_responses = []

        async def mock_streaming_body():
            yield b'data: {"id":"chat_1","choices":[{"delta":{"content":"Partial"}}]}\n\n'
            raise httpx.ReadTimeout("Simulated read timeout mid-stream")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            resp = httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=mock_streaming_body(),
            )
            captured_responses.append(resp)
            return resp

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )

            with pytest.raises(ProviderTimeoutError):
                async for _ in adapter.stream_completion(sample_request):
                    pass

            assert len(captured_responses) == 1
            assert captured_responses[0].is_closed is True
            assert client.is_closed is False

    asyncio.run(_run())


def test_groq_adapter_external_client_reusability(sample_request):
    """Verify an external client remains reusable across multiple serial requests."""
    async def _run():
        sse_text = 'data: {"id":"c","choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=sse_text)

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )

            # Request 1
            chunks1 = [c async for c in adapter.stream_completion(sample_request)]
            assert len(chunks1) == 1
            assert client.is_closed is False

            # Request 2 (reusing same client)
            chunks2 = [c async for c in adapter.stream_completion(sample_request)]
            assert len(chunks2) == 1
            assert client.is_closed is False
            assert call_count == 2

    asyncio.run(_run())


def test_groq_adapter_http_401_auth_error(sample_request):
    """Verify HTTP 401 raises ProviderAuthenticationError."""
    async def _run():
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_invalid_key"),
                http_client=client,
            )
            with pytest.raises(ProviderAuthenticationError, match="HTTP 401 Unauthorized"):
                async for _ in adapter.stream_completion(sample_request):
                    pass

    asyncio.run(_run())


def test_groq_adapter_http_429_burst_rate_limit(sample_request):
    """Verify HTTP 429 burst raises RateLimitError with retry-after header."""
    async def _run():
        err_body = {
            "error": {
                "message": "Rate limit reached on requests per minute (RPM): Limit 30, Used 30. Please try again in 1.85s.",
                "type": "tokens",
            }
        }

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            headers = {"retry-after": "1.85"}
            return httpx.Response(429, headers=headers, json=err_body)

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )
            with pytest.raises(RateLimitError) as exc_info:
                async for _ in adapter.stream_completion(sample_request):
                    pass
            err = exc_info.value
            assert err.retry_after_s == 1.85
            assert err.is_daily is False
            assert err.limit_type == "RPM"

    asyncio.run(_run())


def test_groq_adapter_http_429_daily_budget_exhausted(sample_request):
    """Verify HTTP 429 daily exhaustion raises DailyBudgetExhaustedError."""
    async def _run():
        err_body = {
            "error": {
                "message": "Rate limit reached on requests per day (RPD): Limit 14400, Used 14400.",
                "type": "requests",
            }
        }

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json=err_body)

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )
            with pytest.raises(DailyBudgetExhaustedError):
                async for _ in adapter.stream_completion(sample_request):
                    pass

    asyncio.run(_run())


def test_groq_adapter_http_503_unavailable(sample_request):
    """Verify HTTP 503 raises ProviderUnavailableError."""
    async def _run():
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )
            with pytest.raises(ProviderUnavailableError) as exc_info:
                async for _ in adapter.stream_completion(sample_request):
                    pass
            assert exc_info.value.status_code == 503

    asyncio.run(_run())


def test_groq_adapter_connection_timeout(sample_request):
    """Verify connection timeout raises ProviderTimeoutError."""
    async def _run():
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connection timed out to api.groq.com")

        transport = httpx.MockTransport(mock_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            adapter = GroqProviderAdapter(
                api_key=SecretStr("gsk_dummy_test_key"),
                http_client=client,
            )
            with pytest.raises(ProviderTimeoutError):
                async for _ in adapter.stream_completion(sample_request):
                    pass

    asyncio.run(_run())
