"""Pipeline control-flow test with a scripted client double (tests only, never results)."""
import asyncio
import time

from helpers import FakeChunk, KeywordIndex

from relay.live.groq_client import CallResult
from relay.live.pipeline import LivePipeline, PipelineConfig
from relay.live.ratelimit import TokenBudget
from relay.live.retrieval import LiveRetriever

CHUNKS = [FakeChunk("2410.00037v1_c1", "2410.00037v1", 4, "3.3 Audio Tokenization",
                    "Mimi runs at a frame rate of 12.5 Hz on 24 kHz audio."),
          FakeChunk("1706.03762v7_c1", "1706.03762v7", 5, "3.2.2 Multi-Head Attention", "We use h = 8 heads.")]


class ScriptedClient:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = 0

    async def stream_chat(self, model, messages, max_tokens, temperature=0.0, on_delta=None, budget=None,
                          deadline_at=None, extra_body=None):
        self.calls += 1
        kind, pieces = self.scripts.pop(0)
        r = CallResult(model=model, t_send=time.perf_counter())
        if kind == "429day":
            r.error_kind, r.error_message = "rate_limit_day", "per day"
            if budget:
                budget.observe_429("day", 3600)
            return r
        for p in pieces:
            r.t_first_content = r.t_first_content or time.perf_counter()
            r.text += p
            if on_delta and (await on_delta(p)) is False:
                break
        r.usage = {"prompt_tokens": 300, "completion_tokens": 20, "cached_tokens": 0}
        r.t_done = time.perf_counter()
        return r


class FakeRetriever(LiveRetriever):
    def embed_query(self, text):
        import numpy as np
        v = np.array([1.0, 0.1 * len(text) % 1.0])
        return v / np.linalg.norm(v)


def make(scripts, **kw):
    idx = KeywordIndex(CHUNKS)
    retr = FakeRetriever(CHUNKS, idx, idx)
    client = ScriptedClient(scripts)
    budgets = {"m": TokenBudget(rpm=30, tpm=6000), "fb": TokenBudget(rpm=30, tpm=6000)}
    return LivePipeline(client, retr, budgets, PipelineConfig(model="m", **kw)), client


def run(pipe, q, rid="r1"):
    shown = []

    async def sink(t):
        shown.append(t)
    return asyncio.run(pipe.run(q, "R4_hybrid_route", rid, "test", sink=sink)), shown


def test_answer_then_exact_cache_hit():
    pipe, client = make([("ok", ["Mimi runs at 12.5 Hz [E1]."])])
    rec, shown = run(pipe, "What frame rate does Mimi use?")
    assert rec.status == "ok" and rec.path == "answer" and "12.5 Hz" in rec.answer and rec.first_token_ms is not None
    assert rec.cited_ids == ["2410.00037v1_c1"]
    rec2 = asyncio.run(pipe.run("What frame rate does Mimi use?", "R4_hybrid_route", "r2", "test"))
    assert rec2.path == "exact_cache" and client.calls == 1 and rec2.tokens_in == 0


def test_repair_loop_when_nothing_verifiable():
    pipe, client = make([("ok", ["Mimi runs at 50 Hz."]), ("ok", ["Mimi runs at 12.5 Hz [E1]."])])
    rec, _ = run(pipe, "What frame rate does Mimi use?")
    assert rec.repaired and client.calls == 2 and "12.5" in rec.answer


def test_refusal_path():
    pipe, _ = make([("ok", ["NOT_IN_PAPERS: no GPU details.\n"])])
    rec, _ = run(pipe, "What GPUs trained Moshi?")
    assert rec.path == "model_refusal" and "don't answer" in rec.answer


def test_daily_limit_then_fallback_model():
    pipe, client = make([("429day", []), ("ok", ["Mimi runs at 12.5 Hz [E1]."])], fallback_model="fb")
    rec, _ = run(pipe, "What frame rate does Mimi use?")
    assert rec.status == "rate_limited"
    rec2 = asyncio.run(pipe.run("Mimi frame rate at 24 kHz?", "R4_hybrid_route", "r2", "test"))
    assert rec2.path == "daily_fallback_model" and rec2.model == "fb" and rec2.status == "ok"


def test_daily_exhausted_no_fallback_gives_reset_time_and_excerpts():
    pipe, _ = make([])
    pipe.budgets["m"].mark_daily_exhausted(7200)
    rec, shown = run(pipe, "What frame rate does Mimi use?")
    assert rec.path == "daily_budget" and "frees up again in about" in rec.answer and "arXiv:" in rec.answer
    assert "15:00" not in rec.answer


def test_shed_when_no_slot_before_deadline():
    pipe, _ = make([], deadline_s=0.3, max_in_flight=1)

    async def go():
        await pipe.sem.acquire()  # occupy the only slot
        return await pipe.run("What frame rate does Mimi use?", "R4_hybrid_route", "r1", "t")
    rec = asyncio.run(go())
    assert rec.status == "rate_limited" and rec.path == "shed" and rec.tokens_in == 0
