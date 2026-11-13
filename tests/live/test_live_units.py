import asyncio
import json

from helpers import FakeChunk, KeywordIndex

from relay.live.cache import CachedAnswer, SemanticCache, discriminators
from relay.live.compress import Compressor, Evidence
from relay.live.evals import fact_present, grade_answer, retrieval_metrics
from relay.live.gate import SentenceGate, numbers_in
from relay.live.prompts import REFUSAL_SENTINEL, build_messages, length_policy
from relay.live.ratelimit import HeaderSnapshot, TokenBudget, classify_429, parse_duration
from relay.live.retrieval import LiveRetriever, RetrievalSetting, detect_papers, is_junk_section, repair_sections
from relay.live.sse import iter_sse_payloads, parse_payload


# ---------------- sse
def test_sse_parsing_and_groq_usage():
    lines = ['data: {"choices":[{"delta":{"content":"Hel"}}]}', ": keep-alive", "",
             'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}],'
             '"x_groq":{"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,'
             '"queue_time":0.02,"prompt_time":0.004,"completion_time":0.01,'
             '"prompt_tokens_details":{"cached_tokens":6}}}}', "data: [DONE]"]
    ps = list(iter_sse_payloads(lines))
    d = [parse_payload(p) for p in ps]
    assert d[0].content == "Hel" and d[1].content == "lo" and d[2] is None
    u = d[1].usage
    assert u["cached_tokens"] == 6 and abs(u["queue_time_ms"] - 20.0) < 1e-9 and d[1].finish_reason == "stop"


def test_reasoning_delta_kept_separate():
    d = parse_payload('{"choices":[{"delta":{"reasoning":"think"}}]}')
    assert d.content == "" and d.reasoning == "think"


# ---------------- rate limits
def test_parse_duration_formats():
    assert parse_duration("7.66s") == 7.66
    assert abs(parse_duration("2m59.56s") - 179.56) < 1e-9
    assert parse_duration("3h15m") == 11700
    assert parse_duration("120ms") == 0.12
    assert parse_duration("1.85") == 1.85 and parse_duration(None) is None


def test_headers_and_429_classification():
    s = HeaderSnapshot.from_headers({"X-RateLimit-Remaining-Tokens": "512", "x-ratelimit-reset-tokens": "4.2s",
                                     "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2h"}, 100.0)
    assert s.tpm_remaining == 512 and s.tpm_reset_s == 4.2 and s.rpd_remaining == 0
    assert classify_429("... on requests per day (RPD): Limit 14400 ... try again in 3h15m")[0] == "day"
    k, ra = classify_429("Rate limit reached on tokens per minute (TPM): Limit 6000. Please try again in 1.85s.")
    assert k == "minute" and ra == 1.85


def test_budget_bucket_refills_continuously_and_refunds():
    now = [0.0]
    b = TokenBudget(rpm=30, tpm=1200, safety=1.0, clock=lambda: now[0])   # refills 20 tokens/s
    e = b.reserve(1100)
    assert abs(b.wait_time(200) - 5.0) < 1e-9   # need 100 more tokens at 20/s
    now[0] = 5.0
    assert b.wait_time(200) == 0
    b.settle(e, 300)                            # real usage smaller -> refund 800
    assert b.available_tokens() == 1000          # 100 left + 100 refilled + 800 refunded
    assert b.wait_time(5000) == float("inf")


def test_budget_provider_headers_only_lower_estimate():
    now = [0.0]
    b = TokenBudget(rpm=30, tpm=8000, safety=1.0, clock=lambda: now[0])
    b.observe(HeaderSnapshot(observed_at=0, tpm_limit=8000, tpm_remaining=6345, tpm_reset_s=12.41))
    assert b.available_tokens() == 6345
    b.observe(HeaderSnapshot(observed_at=0, tpm_limit=8000, tpm_remaining=7900))
    assert b.available_tokens() == 6345          # never raised by headers


def test_budget_request_rate():
    now = [0.0]
    b = TokenBudget(rpm=30, tpm=10**6, safety=1.0, clock=lambda: now[0])
    for _ in range(30):
        b.reserve(1)
    assert abs(b.wait_time(1) - 2.0) < 1e-9      # 30 RPM -> one request every 2 s


def test_budget_daily_from_headers_and_local_cap():
    b = TokenBudget(rpm=30, tpm=6000, local_daily_request_cap=2)
    b.reserve(10); b.reserve(10)
    assert b.daily_blocked()
    b2 = TokenBudget(rpm=30, tpm=6000)
    b2.observe(HeaderSnapshot(observed_at=0, rpd_remaining=0, rpd_reset_s=3600))
    assert b2.daily_blocked() and b2.daily_reset_s == 3600


def test_acquire_sheds_when_deadline_too_short():
    async def go():
        b = TokenBudget(rpm=30, tpm=1000, safety=1.0)
        b.reserve(1000)
        import time
        ok, waited, entry = await b.acquire(100, time.monotonic() + 2.0)
        return ok, entry
    ok, entry = asyncio.run(go())
    assert ok is False and entry is None


# ---------------- gate
def ev(tag, pid, page, sec, text):
    c = FakeChunk(chunk_id=f"{pid}_c{tag}", paper_id=pid, page=page, section=sec, text=text)
    return Evidence(tag=tag, chunk=c, text=text, tokens_before=10, tokens_after=10)


EVS = [ev("E1", "2410.00037v1", 4, "3.3 Audio Tokenization", "Mimi runs at a frame rate of 12.5 Hz, from 24 kHz audio."),
       ev("E2", "1706.03762v7", 5, "3.2.2 Multi-Head Attention", "We use h = 8 heads and d_k = d_v = 64.")]


def test_gate_streams_only_verified_sentences():
    g = SentenceGate(EVS, "What frame rate?")
    out = []
    for piece in ["Mimi runs at 12.", "5 Hz [E1]. It was", " trained on 7 GPUs [E1]. Heads: 8 [E2]."]:
        out += g.feed(piece)
    out += g.finish()
    assert len(out) == 2 and "12.5 Hz" in out[0] and "arXiv:2410.00037v1 p.4" in out[0]
    assert "8" in out[1] and g.violations == 1  # "7 GPUs" is not in E1 -> blocked
    assert g.rejected[0].missing_numbers == ["7"]


def test_gate_citation_after_period_and_invalid_tag():
    g = SentenceGate(EVS)
    out = g.feed("Mimi is 12.5 Hz. [E1] Also 64 [E9]. ") + g.finish()
    assert len(out) == 1 and "arXiv:2410.00037v1" in out[0]
    assert any(v.kind == "invalid_citation" for v in g.rejected)


def test_gate_refusal_and_lead_in():
    g = SentenceGate(EVS)
    g.feed(f"{REFUSAL_SENTINEL}: the passages do not mention GPUs.\n")
    g.finish()
    assert g.refusal and not g.emitted
    g2 = SentenceGate(EVS)
    out = g2.feed("Here is what the paper says. Mimi uses 12.5 Hz [E1]. ") + g2.finish()
    assert len(out) == 2 and out[0].startswith("Here is")


def test_numbers_ignore_citation_tags():
    assert numbers_in("value 512 [E3] and [E1, E2]") == {"512"}
    assert numbers_in("1,920 and 6.0") == {"1920", "6"}


# ---------------- caches
def test_discriminators_block_base_vs_big():
    assert discriminators("d_model of the base Transformer") != discriminators("d_model of the big Transformer")


def test_semantic_cache_requires_discriminators_and_evidence():
    import numpy as np
    sc = SemanticCache(threshold=0.9)
    v = np.array([1.0, 0.0]); v2 = np.array([0.99, 0.141])
    sc.add("frame rate of mimi", v, CachedAnswer("frame rate of mimi", "12.5 Hz", ["a", "b"], ["a"], "m", "s"))
    assert sc.lookup("mimi frame rate?", v2, ["a", "c"], "s", "m") is not None
    assert sc.lookup("mimi frame rate?", v2, ["x", "y"], "s", "m") is None           # evidence changed
    assert sc.lookup("frame rate of mimi at 24 khz", v2, ["a"], "s", "m") is None      # new number
    assert sc.stats["blocked_by_evidence"] == 1 and sc.stats["blocked_by_discriminator"] == 1


# ---------------- compression / prompts
def test_compression_keeps_answer_sentence_and_dedupes():
    filler = " ".join(f"Sentence {i} talks about unrelated training details." for i in range(60))
    c1 = FakeChunk("c1", "p", 1, "s", filler + " Mimi operates at 12.5 Hz frame rate. " + filler)
    c2 = FakeChunk("c2", "p", 1, "s", c1.text)  # duplicate window
    comp = Compressor([c1.text, "other doc about attention"])
    evs = comp.build("What frame rate does Mimi use?", [c1, c2], per_chunk_tokens=80)
    assert len(evs) == 1 and "12.5 Hz" in evs[0].text and evs[0].tokens_after < evs[0].tokens_before


def test_length_policy_and_message_layout():
    p = length_policy("What is d_model?")
    assert p.max_tokens == 120
    assert length_policy("How does Moshi differ from prior systems?").label == "explain"
    assert length_policy("What is d_model?", pressure=0.9).max_tokens < 120
    m = build_messages("Q?", EVS, "lean", p)
    assert m[1]["content"].rstrip().endswith("ANSWER:") and "QUESTION: Q?" in m[1]["content"]
    assert build_messages("Q2?", EVS, "lean", p)[0] == m[0]  # static system prompt -> cacheable prefix


# ---------------- retrieval
def test_junk_sections_and_repair():
    assert is_junk_section("0 10 20 30 40 50") and is_junk_section("8 4.88 25.5 80")
    assert not is_junk_section("3.2.2 Multi-HeadAttention") and not is_junk_section("Abstract")
    cs = [FakeChunk("a_p01", "a", 1, "2 Method", "x"), FakeChunk("a_p02", "a", 2, "0 10 20", "y")]
    fixed = repair_sections(cs)
    assert fixed[1].section == "2 Method" and fixed[1].metadata["section_repaired_from"] == "0 10 20"


def test_routing_and_parallel_fanout():
    assert detect_papers("What frame rate does Mimi use?") == ["2410.00037v1"]
    cs = [FakeChunk("m1", "2410.00037v1", 4, "Audio", "mimi frame rate 12.5 hz transformer"),
          FakeChunk("t1", "1706.03762v7", 5, "Attention", "transformer heads frame rate"),
          FakeChunk("d1", "2507.07955v2", 3, "Chunking", "dynamic chunking frame rate transformer")]
    idx = KeywordIndex(cs)
    r = LiveRetriever(cs, idx, idx)
    res = asyncio.run(r.retrieve("frame rate in moshi transformer", RetrievalSetting("t", top_k=2)))
    assert all(c.paper_id == "2410.00037v1" for c in res.chunks)
    res2 = asyncio.run(r.retrieve("Compare Moshi and Attention Is All You Need transformer", RetrievalSetting("t", top_k=2)))
    assert {c.paper_id for c in res2.chunks} == {"2410.00037v1", "1706.03762v7"} and len(res2.subqueries) == 2


# ---------------- evals
def test_fact_matching_and_grades():
    assert fact_present("12.5 Hz", "runs at 12.5hz") and fact_present("h = 8", "uses h=8 heads")
    assert not fact_present("512", "dimension 5120")
    q = {"question": "x", "is_unanswerable": False, "key_facts": ["512", "h = 8", "64"], "target_paper_id": "1706.03762v7"}
    good = grade_answer(q, "d_model 512, h = 8, d_k 64", ["1706.03762v7_c1"], "ok")
    assert good.grade == "correct"
    assert grade_answer(q, "d_model 512", ["1706.03762v7_c1"], "ok").grade == "wrong"
    assert grade_answer(q, "The papers don't answer this: x", [], "ok").grade == "refused_wrongly"
    assert grade_answer(q, "", [], "rate_limited").grade == ""
    qu = {"question": "x", "is_unanswerable": True, "key_facts": []}
    assert grade_answer(qu, "The papers don't answer this: no", [], "ok").grade == "refused_correctly"


def test_retrieval_metrics():
    q = {"target_paper_id": "p", "target_page": 4, "key_facts": ["12.5 Hz"], "is_unanswerable": False}
    cs = [FakeChunk("x", "q", 1, "s", "nothing"), FakeChunk("y", "p", 5, "s", "Mimi at 12.5 Hz")]
    m = retrieval_metrics(q, cs, compressed_text="Mimi at 12.5 Hz")
    assert m["hit_at_k"] == 1 and m["mrr"] == 0.5 and m["paper_precision"] == 0.5 and m["fact_retention"] == 1.0


def test_results_writer_and_tables_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace as N

    from relay.live.records import BRIEF_COLUMNS, ResultsWriter
    from relay.live.tables import main as tables_main, peak_queue

    d = Path(tmp_path or tempfile.mkdtemp())
    w = ResultsWriter(d / "r.csv", d / "t.jsonl")
    base = dict(source="real", retries=0, tokens_in=1000, tokens_out=50, setting_label="S", model="llama-3.1-8b-instant",
                cached_tokens=0, provider_first_token_ms=300.0)
    rows = [("steady", "2026-01-01T00:00:00.000+00:00", "ok", 400.0, 900.0, 0.0),
            ("steady", "2026-01-01T00:00:30.000+00:00", "ok", 500.0, 1000.0, 2000.0),
            ("steady", "2026-01-01T00:00:31.000+00:00", "rate_limited", 5.0, 6.0, 1500.0),
            ("retrieval_eval", "2026-01-01T00:01:00.000+00:00", "ok", 450.0, 950.0, 0.0)]
    for i, (sc, st, status, ft, tt, qw) in enumerate(rows):
        rec = N(request_id=f"r{i}", scenario=sc, start_time=st, first_token_ms=ft, total_ms=tt, status=status,
                path="answer", queue_wait_ms=qw, to_trace=lambda: {}, **base)
        w.write(rec, "correct" if sc == "retrieval_eval" else "")
    w.close()
    header = (d / "r.csv").read_text().splitlines()[0].split(",")
    assert header[:12] == BRIEF_COLUMNS
    out = tables_main(str(d / "r.csv"))
    assert "| steady | real | 3 | 31.0 |" in out and "66.7 | 33.3 | 0.0" in out
    assert "| S | 1 (graded 1) | 100.0 |" in out
    import pandas as pd
    assert peak_queue(pd.read_csv(d / "r.csv")[lambda x: x.scenario == "steady"]) == 2


def test_numbers_glued_to_letters_are_grounded():
    e = [ev("E1", "1706.03762v7", 8, "6.1 MachineTranslation", "newstate-of-the-artBLEUscoreof28.4. Thec")]
    g = SentenceGate(e)
    out = g.feed("The big model scored 28.4 BLEU [E1]. ") + g.finish()
    assert len(out) == 1 and g.violations == 0


def test_query_expansion_and_numeric_noise():
    from relay.live.retrieval import expand_query, is_numeric_noise
    q = expand_query("what is d_model and d_k?")
    assert "dmodel" in q and "d model" in q and "dk" in q
    assert is_numeric_noise(FakeChunk("x", "p", 9, "s", "1 512 512 5.29 24.9 4 128 128 5.00 25.5"))
    assert not is_numeric_noise(FakeChunk("t", "p", 9, "s", "| 6 512 2048 |", content_type="table"))
    assert not is_numeric_noise(FakeChunk("y", "p", 3, "s", "We use dmodel = 512 for the base model."))


def test_partial_answer_keeps_missing_note():
    g = SentenceGate(EVS)
    g.feed("Mimi runs at 12.5 Hz [E1].\n" + REFUSAL_SENTINEL + ": the downsampling factor is not given.\n")
    g.finish()
    a = g.answer_text()
    assert "12.5 Hz" in a and "papers don't state" in a and g.refusal is None


def test_judge_primary_grading():
    q = {"question": "x", "is_unanswerable": False, "key_facts": ["12.5 Hz", "512", "1920", "24 kHz"],
         "target_paper_id": "2410.00037v1"}
    ans = "Mimi runs at 12.5 Hz with latent dimension 512"
    assert grade_answer(q, ans, ["2410.00037v1_c1"], "ok", policy="rules").grade == "wrong"
    assert grade_answer(q, ans, ["2410.00037v1_c1"], "ok", judge={"correct": True}).grade == "correct"
    assert grade_answer(q, ans, ["1706.03762v7_c1"], "ok", judge={"correct": True}).grade == "wrong"  # wrong paper


def test_daily_block_reopens_after_retry_after():
    now = [0.0]
    b = TokenBudget(rpm=30, tpm=8000, clock=lambda: now[0])
    b.observe_429("day", 419.0)          # real capture: "try again in 6m58.608s", retry-after: 419
    assert b.daily_blocked()
    now[0] = 418.0
    assert b.daily_blocked()
    now[0] = 420.0
    assert not b.daily_blocked()          # Groq refilled enough: try again
    c = TokenBudget(rpm=30, tpm=8000, local_daily_request_cap=1, clock=lambda: now[0])
    c.reserve(1)
    now[0] = 10**6
    assert c.daily_blocked()              # local test cap never re-opens
