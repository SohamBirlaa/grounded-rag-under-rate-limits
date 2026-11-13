"""Unit tests for the deterministic controllable stand-in inference engine."""

import asyncio
import pytest

from relay.config.settings import StandInConfig
from relay.contracts.errors import (
    DailyBudgetExhaustedError,
    ProviderAuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from relay.providers.base import ProviderCompletionRequest
from relay.providers.standin import StandInProviderAdapter


@pytest.fixture
def rag_prompt_with_passage():
    return (
        "Here are the relevant passages:\n\n"
        '<passage id="1706.03762v7_sec_3_2_p04_c014" paper_id="1706.03762v7" '
        'version="v7" page="4" section="3.2.1 Scaled Dot-Product Attention" content_type="prose">\n'
        "We compute the attention function on a set of queries simultaneously, "
        "packed together into a matrix Q.\n"
        "</passage>\n\n"
        "Question: What is Scaled Dot-Product Attention?"
    )


def test_standin_deterministic_grounded_answer(rag_prompt_with_passage):
    """Verify stand-in generates a deterministic grounded answer citing input passages."""
    async def _run():
        standin = StandInProviderAdapter(inject_delay=False)
        request = ProviderCompletionRequest(
            messages=[{"role": "user", "content": rag_prompt_with_passage}],
            model_id="llama-3.1-8b-instant-standin",
        )

        chunks = []
        async for chunk in standin.stream_completion(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[0].is_first_token is True

        assembled_text = "".join(c.text_delta for c in chunks)
        assert "arXiv:1706.03762v7" in assembled_text
        assert "1706.03762v7_sec_3_2_p04_c014" in assembled_text
        assert "attention function" in assembled_text

        # Verify usage counters
        final_chunk = chunks[-1]
        assert final_chunk.prompt_tokens is not None and final_chunk.prompt_tokens > 0
        assert final_chunk.completion_tokens is not None and final_chunk.completion_tokens > 0

    asyncio.run(_run())


def test_standin_epistemic_abstention_on_empty_context():
    """Verify stand-in abstains when prompt indicates no passages were retrieved."""
    async def _run():
        empty_prompt = (
            "<retrieved_passages>\n"
            "  <!-- No relevant passages retrieved -->\n"
            "</retrieved_passages>\n\n"
            "Question: What is Llama-3 dataset mixture?"
        )
        standin = StandInProviderAdapter(inject_delay=False)
        request = ProviderCompletionRequest(
            messages=[{"role": "user", "content": empty_prompt}],
        )

        chunks = []
        async for chunk in standin.stream_completion(request):
            chunks.append(chunk)

        assembled_text = "".join(c.text_delta for c in chunks)
        assert "cannot answer this question based on the provided corpus" in assembled_text

    asyncio.run(_run())


def test_standin_failure_modes():
    """Verify controllable failure injection across standard error types."""
    async def _run():
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": "Test prompt"}],
        )

        # 1. Burst rate limit
        standin_rl = StandInProviderAdapter(inject_delay=False, failure_mode="rate_limit_burst")
        with pytest.raises(RateLimitError):
            async for _ in standin_rl.stream_completion(req):
                pass

        # 2. Daily exhaustion
        standin_daily = StandInProviderAdapter(inject_delay=False, failure_mode="daily_exhausted")
        with pytest.raises(DailyBudgetExhaustedError):
            async for _ in standin_daily.stream_completion(req):
                pass

        # 3. Auth error
        standin_auth = StandInProviderAdapter(inject_delay=False, failure_mode="auth_error")
        with pytest.raises(ProviderAuthenticationError):
            async for _ in standin_auth.stream_completion(req):
                pass

        # 4. Server error
        standin_503 = StandInProviderAdapter(inject_delay=False, failure_mode="server_error")
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in standin_503.stream_completion(req):
                pass
        assert exc_info.value.status_code == 503

        # 5. Timeout
        standin_to = StandInProviderAdapter(inject_delay=False, failure_mode="timeout")
        with pytest.raises(ProviderTimeoutError):
            async for _ in standin_to.stream_completion(req):
                pass

    asyncio.run(_run())


def test_standin_fail_n_attempts_then_succeed(rag_prompt_with_passage):
    """Verify stand-in fails for configured number of attempts then succeeds."""
    async def _run():
        standin = StandInProviderAdapter(
            inject_delay=False,
            failure_mode="server_error",
            fail_attempts=2,
        )
        req = ProviderCompletionRequest(
            messages=[{"role": "user", "content": rag_prompt_with_passage}],
        )

        # Attempt 1: Fails
        with pytest.raises(ProviderUnavailableError):
            async for _ in standin.stream_completion(req):
                pass

        # Attempt 2: Fails
        with pytest.raises(ProviderUnavailableError):
            async for _ in standin.stream_completion(req):
                pass

        # Attempt 3: Succeeds!
        chunks = []
        async for c in standin.stream_completion(req):
            chunks.append(c)
        assert len(chunks) > 0
        assert "attention function" in "".join(c.text_delta for c in chunks)

    asyncio.run(_run())


def test_standin_quota_snapshot():
    """Verify stand-in initializes valid quota snapshot."""
    cfg = StandInConfig(daily_rpd=14400, rate_limit_rpm=30, rate_limit_tpm=6000)
    standin = StandInProviderAdapter(config=cfg, inject_delay=False)
    snapshot = standin.get_quota_snapshot()
    assert snapshot.rpd_limit == 14400
    assert snapshot.rpm_limit == 30
    assert snapshot.tpm_limit == 6000
    assert snapshot.is_daily_exhausted() is False
