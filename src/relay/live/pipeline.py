"""The live Relay pipeline (real Groq only).

Steps (>=4, with a tool call, branches and a bounded loop):
  1. understand   - route to papers, pick an output-length policy (no LLM call: saves quota)
  2. cache        - exact-match cache
  3. retrieve     - `retrieve_passages` tool: hybrid search, routing, re-rank, parallel fan-out
  4. decide       - branch: abstain (low evidence) | semantic-cache hit | daily budget gone |
                    compact mode under TPM pressure | full answer
  5. generate     - streamed from Groq through the sentence gate (only verified sentences are shown)
  6. verify/repair- loop: at most one repair call if nothing verifiable came back
  7. record       - one results.csv row + one trace line with every timing span
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from relay.live.cache import CachedAnswer, ExactCache, SemanticCache, cache_key
from relay.live.compress import Compressor, Evidence
from relay.live.gate import SentenceGate
from relay.live.groq_client import REASONING_MODELS_PREFIX, CallResult, GroqClient
from relay.live.prompts import build_messages, build_repair_messages, estimate_prompt_tokens, length_policy
from relay.live.ratelimit import TokenBudget
from relay.live.retrieval import SETTINGS, LegacySettingAdapter, LiveRetriever, RetrievalResult, RetrievalSetting

TextSink = Callable[[str], Awaitable[None]]

LEGACY_SETTINGS = {
    "S1_256_3_no": RetrievalSetting(LegacySettingAdapter.LABELS["S1_256_3_no"], top_k=3, compress=False,
                                    route_papers=False, filter_junk=False, fanout=False),
    "S3_450_8to4_rrf_no": RetrievalSetting(LegacySettingAdapter.LABELS["S3_450_8to4_rrf_no"], top_k=4,
                                           compress=False, route_papers=False, filter_junk=False, fanout=False),
}


@dataclass
class PipelineConfig:
    model: str = "openai/gpt-oss-20b"
    fallback_model: Optional[str] = None  # used when the primary model's daily budget is gone
    # Measured: 0 cached tokens on 7 identical ~1.3-2.5K-token prompts to gpt-oss-20b (free tier), full
    # tokens charged. Few-shot "cache_friendly" prompt therefore only costs TPM -> default lean.
    prompt_profile: str = "lean"          # lean | cache_friendly | auto
    deadline_s: float = 12.0              # how long a person waits before giving up
    max_in_flight: int = 4
    token_ratio: float = 1.25             # real/local tokens, measured by live_probe.py (gpt-oss-20b)
    answerability_threshold: Optional[float] = None  # cross-encoder score below which we refuse without an LLM call
    use_exact_cache: bool = True
    use_semantic_cache: bool = True
    semantic_threshold: float = 0.92
    repair: bool = True
    corpus_version: str = "s23_450+repaired_sections_v1"


@dataclass
class RunRecord:
    request_id: str
    scenario: str
    question: str
    setting_key: str
    setting_label: str
    model: str
    start_time: str
    source: str = "real"
    status: str = "ok"             # ok | rate_limited | timeout | error
    path: str = ""                 # answer | exact_cache | semantic_cache | abstain | model_refusal |
                                   # unverified_refusal | shed | daily_budget | daily_fallback_model
    answer: str = ""
    first_token_ms: Optional[float] = None      # first text the PERSON sees (after gate), from arrival
    provider_first_token_ms: Optional[float] = None  # first raw Groq content token, from arrival
    total_ms: float = 0.0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    queue_wait_ms: float = 0.0     # waiting for a slot / for TPM budget (ours)
    retrieval_ms: float = 0.0
    compress_ms: float = 0.0
    generation_ms: float = 0.0
    repair_ms: float = 0.0
    retry_wait_ms: float = 0.0
    groq_queue_ms: Optional[float] = None       # Groq-reported server-side queue time
    groq_prompt_ms: Optional[float] = None
    groq_completion_ms: Optional[float] = None
    evidence_ids: List[str] = field(default_factory=list)
    cited_ids: List[str] = field(default_factory=list)
    gate_problems: List[str] = field(default_factory=list)
    context_tokens_before: int = 0
    context_tokens_after: int = 0
    max_tokens: int = 0
    finish_reason: Optional[str] = None
    routed_papers: List[str] = field(default_factory=list)
    top_rerank_score: Optional[float] = None
    repaired: bool = False
    gate_emitted: int = 0
    gate_rejected: int = 0
    pressure: float = 0.0
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    semantic_similarity: Optional[float] = None

    def to_trace(self) -> Dict[str, Any]:
        return asdict(self)


class LivePipeline:
    def __init__(
        self,
        client: GroqClient,
        retriever: LiveRetriever,
        budgets: Dict[str, TokenBudget],
        cfg: PipelineConfig,
        legacy: Optional[LegacySettingAdapter] = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.legacy = legacy
        self.budgets = budgets
        self.cfg = cfg
        self.compressor = Compressor(c.text for c in retriever.chunks)
        self.exact = ExactCache()
        self.semantic = SemanticCache(threshold=cfg.semantic_threshold)
        self.sem = asyncio.Semaphore(cfg.max_in_flight)
        self.waiting = 0
        self.peak_waiting = 0
        self.in_flight = 0

    # ------------------------------------------------------------------ helpers
    def _setting(self, key: str) -> RetrievalSetting:
        return SETTINGS[key] if key in SETTINGS else LEGACY_SETTINGS[key]

    async def retrieve_passages(self, question: str, key: str) -> RetrievalResult:
        """The tool call."""
        if key in LEGACY_SETTINGS:
            if self.legacy is None:
                raise RuntimeError("legacy settings need a LegacySettingAdapter")
            return await self.legacy.retrieve(question, key)
        return await self.retriever.retrieve(question, SETTINGS[key])

    def _profile(self, model: str) -> str:
        if self.cfg.prompt_profile != "auto":
            return self.cfg.prompt_profile
        return "cache_friendly" if model.startswith(REASONING_MODELS_PREFIX) else "lean"

    @staticmethod
    def _excerpts(chunks: List[Any], n: int = 2, chars: int = 220) -> str:
        lines = []
        for c in chunks[:n]:
            snippet = " ".join((c.text or "").split())[:chars]
            lines.append(f"- [arXiv:{c.paper_id} p.{c.page} §{c.section}] \"{snippet}…\"")
        return "\n".join(lines)

    def _reset_text(self, budget: TokenBudget) -> str:
        if budget.daily_reset_s is None or budget.daily_exhausted_at is None:
            return "gradually as the provider refills its daily allowance"
        remaining = budget.daily_reset_s - (time.monotonic() - budget.daily_exhausted_at)
        at = datetime.now(timezone.utc) + timedelta(seconds=max(0.0, remaining))
        h, m = divmod(int(max(0, remaining)) // 60, 60)
        return f"in about {h}h {m:02d}m (≈{at:%H:%M} UTC, from Groq's retry-after; capacity refills continuously)"

    # ------------------------------------------------------------------ one model call through the gate
    async def _generate(self, rec: RunRecord, model: str, messages: List[Dict[str, str]], max_tokens: int,
                        evidence: List[Evidence], question: str, t0: float, deadline: float,
                        sink: Optional[TextSink]) -> tuple[SentenceGate, CallResult]:
        gate = SentenceGate(evidence, question)

        async def emit(lines: List[str]) -> None:
            for line in lines:
                if rec.first_token_ms is None:
                    rec.first_token_ms = (time.perf_counter() - t0) * 1000.0
                if sink:
                    await sink(line + " ")

        async def on_delta(text: str) -> bool:
            await emit(gate.feed(text))
            return not gate.should_abort()

        res = await self.client.stream_chat(model, messages, max_tokens, on_delta=on_delta,
                                            budget=self.budgets[model], deadline_at=deadline)
        await emit(gate.finish())
        rec.gate_emitted, rec.gate_rejected = len(gate.emitted), len(gate.rejected)
        u = res.usage or {}
        rec.tokens_in += u.get("prompt_tokens", 0)
        rec.tokens_out += u.get("completion_tokens", 0)
        rec.cached_tokens += u.get("cached_tokens", 0)
        rec.reasoning_tokens += u.get("reasoning_tokens", 0)
        rec.retries += res.retries
        rec.retry_wait_ms += res.retry_wait_s * 1000.0
        rec.finish_reason = res.finish_reason
        rec.attempts.extend(res.attempts)
        if res.t_first_content and rec.provider_first_token_ms is None:
            rec.provider_first_token_ms = (res.t_first_content - t0) * 1000.0
        for k in ("queue_time_ms", "prompt_time_ms", "completion_time_ms"):
            if u.get(k) is not None:
                setattr(rec, "groq_" + k.replace("_time_ms", "_ms"), u[k])
        return gate, res

    async def _admit(self, model: str, est: int, deadline_mono: float):
        self.waiting += 1
        self.peak_waiting = max(self.peak_waiting, self.waiting)
        try:
            try:
                await asyncio.wait_for(self.sem.acquire(), timeout=max(0.0, deadline_mono - time.monotonic()))
            except asyncio.TimeoutError:
                return False, 0.0, None  # no slot freed before the person would give up -> shed
            ok, waited, entry = await self.budgets[model].acquire(est, deadline_mono)
            if not ok:
                self.sem.release()
            return ok, waited, entry
        finally:
            self.waiting -= 1

    # ------------------------------------------------------------------ main entry
    async def run(self, question: str, setting_key: str, request_id: str, scenario: str,
                  sink: Optional[TextSink] = None) -> RunRecord:
        t0 = time.perf_counter()
        deadline = t0 + self.cfg.deadline_s
        deadline_mono = time.monotonic() + self.cfg.deadline_s
        setting = self._setting(setting_key)
        model = self.cfg.model
        rec = RunRecord(request_id=request_id, scenario=scenario, question=question, setting_key=setting_key,
                        setting_label=setting.name, model=model,
                        start_time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))

        async def say(text: str) -> None:
            if rec.first_token_ms is None:
                rec.first_token_ms = (time.perf_counter() - t0) * 1000.0
            rec.answer += text
            if sink:
                await sink(text)

        try:
            # 2. exact cache
            key = cache_key(question, setting.name, model, self.cfg.corpus_version)
            if self.cfg.use_exact_cache and (hit := self.exact.get(key)):
                rec.path, rec.cited_ids = "exact_cache", hit.cited_ids
                await say(hit.answer)
                return self._done(rec, t0)

            # 3. retrieve (tool call)
            rr = await self.retrieve_passages(question, setting_key)
            rec.retrieval_ms = rr.ms
            rec.evidence_ids = [c.chunk_id for c in rr.chunks]
            rec.routed_papers, rec.top_rerank_score = rr.routed_papers, rr.top_rerank_score

            # 4. decide
            if (self.cfg.answerability_threshold is not None and rr.top_rerank_score is not None
                    and rr.top_rerank_score < self.cfg.answerability_threshold):
                rec.path = "abstain"
                await say("The three papers do not appear to cover this question (no passage scored as relevant), "
                          "so I won't guess.")
                return self._done(rec, t0)

            qvec = None
            if self.cfg.use_semantic_cache:
                qvec = await asyncio.to_thread(self.retriever.embed_query, question)
                sh = self.semantic.lookup(question, qvec, rec.evidence_ids, setting.name, model)
                if sh:
                    rec.path, rec.semantic_similarity = "semantic_cache", sh["similarity"]
                    rec.cited_ids = sh["item"].cited_ids
                    await say(sh["item"].answer)
                    return self._done(rec, t0)

            primary_budget = self.budgets[model]
            if primary_budget.daily_blocked():
                fb = self.cfg.fallback_model
                if fb and fb in self.budgets and not self.budgets[fb].daily_blocked():
                    model = fb
                    rec.model, rec.path = fb, "daily_fallback_model"
                else:
                    rec.status, rec.path = "rate_limited", "daily_budget"
                    await say("Live answers are paused: today's model budget is used up. It frees up again "
                              f"{self._reset_text(primary_budget)}; after that answers return slowly (about one every "
                              "10 minutes, as Groq refills ~2.3 tokens/s). Please ask again then. Meanwhile, these are "
                              "the most relevant passages (not checked by the model):\n" + self._excerpts(rr.chunks))
                    return self._done(rec, t0)

            budget = self.budgets[model]
            rec.pressure = round(budget.pressure(), 3)
            compact = rec.pressure >= 0.75
            tc = time.perf_counter()
            chunks = rr.chunks[:2] if compact else rr.chunks
            evidence = self.compressor.build(
                question, chunks, compress=setting.compress or compact,
                per_chunk_tokens=140 if compact else 220, total_tokens=500 if compact else 900)
            rec.compress_ms = (time.perf_counter() - tc) * 1000.0
            rec.context_tokens_before = sum(e.tokens_before for e in evidence)
            rec.context_tokens_after = sum(e.tokens_after for e in evidence)

            policy = length_policy(question, rec.pressure, model.startswith(REASONING_MODELS_PREFIX))
            rec.max_tokens = policy.max_tokens
            messages = build_messages(question, evidence, self._profile(model), policy)
            est = estimate_prompt_tokens(messages, self.cfg.token_ratio) + policy.max_tokens

            # admission: in-flight slot + per-minute budget
            tq = time.perf_counter()
            ok, _waited, entry = await self._admit(model, est, deadline_mono)
            rec.queue_wait_ms = (time.perf_counter() - tq) * 1000.0
            if not ok:
                rec.status = "rate_limited"
                if budget.daily_blocked():
                    rec.path = "daily_budget"
                    await say(f"Live answers are paused: today's model budget is used up. It frees up again "
                              f"{self._reset_text(budget)}.\n" + self._excerpts(rr.chunks))
                else:
                    rec.path = "shed"
                    await say("We're at capacity right now and couldn't start your answer within "
                              f"{self.cfg.deadline_s:.0f}s. Please retry in a minute.")
                return self._done(rec, t0)

            try:
                self.in_flight += 1
                tg = time.perf_counter()
                gate, res = await self._generate(rec, model, messages, policy.max_tokens, evidence, question,
                                                 t0, deadline, sink)
                rec.generation_ms = (time.perf_counter() - tg) * 1000.0
                budget.settle(entry, (res.usage.get("prompt_tokens", 0) - res.usage.get("cached_tokens", 0)
                                      + res.usage.get("completion_tokens", 0)) if res.usage else est)

                # 6. verify / repair loop (bounded: one repair)
                if (res.ok and not gate.emitted and gate.refusal is None and self.cfg.repair
                        and time.perf_counter() + 3.0 < deadline):
                    rec.gate_problems = gate.problems()
                    rmsgs = build_repair_messages(messages, res.text, rec.gate_problems or ["no citations"])
                    rest = estimate_prompt_tokens(rmsgs, self.cfg.token_ratio) + policy.max_tokens
                    ok2, _w2, entry2 = await budget.acquire(rest, deadline_mono - 2.0)
                    if ok2:
                        tr = time.perf_counter()
                        gate, res2 = await self._generate(rec, model, rmsgs, policy.max_tokens, evidence, question,
                                                          t0, deadline, sink)
                        rec.repair_ms = (time.perf_counter() - tr) * 1000.0
                        rec.repaired = True
                        u2 = res2.usage or {}
                        budget.settle(entry2, u2.get("prompt_tokens", 0) - u2.get("cached_tokens", 0)
                                      + u2.get("completion_tokens", 0) if u2 else rest)
                        res = res2
                rec.gate_problems = rec.gate_problems or gate.problems()
            finally:
                self.in_flight -= 1
                self.sem.release()

            # 7. outcome
            if gate.emitted:
                rec.answer = gate.answer_text()
                rec.cited_ids = gate.cited_chunks()
                rec.path = rec.path or "answer"
                if res.error_kind == "timeout":
                    rec.status = "timeout"  # partial, verified-only answer was shown
                elif res.ok:
                    item = CachedAnswer(question, rec.answer, rec.evidence_ids, rec.cited_ids, model, setting.name)
                    if self.cfg.use_exact_cache:
                        self.exact.put(key, item)
                    if qvec is not None:
                        self.semantic.add(question, qvec, item)
            elif gate.refusal is not None:
                rec.path = rec.path or "model_refusal"
                await say(f"The papers don't answer this: {gate.refusal}")
            elif res.error_kind in ("rate_limit_minute", "rate_limit_day"):
                rec.status, rec.path = "rate_limited", ("daily_budget" if res.error_kind == "rate_limit_day" else "shed")
                rec.error = res.error_message
                await say("The model service is rate-limiting us right now; please retry shortly.")
            elif res.error_kind == "timeout":
                rec.status, rec.path, rec.error = "timeout", "timeout", res.error_message
                await say("This is taking too long, so I stopped rather than keep you waiting. Please retry.")
            elif res.error_kind:
                rec.status, rec.path, rec.error = "error", "error", res.error_message
                await say("Something went wrong talking to the model service.")
            else:
                rec.path = "unverified_refusal"
                await say("I couldn't produce an answer I can verify against the papers, so I won't guess. "
                          "Closest passages:\n" + self._excerpts(rr.chunks))
            return self._done(rec, t0)
        except Exception as e:  # never crash a benchmark run
            rec.status, rec.path, rec.error = "error", "error", f"{type(e).__name__}: {e}"
            return self._done(rec, t0)

    @staticmethod
    def _done(rec: RunRecord, t0: float) -> RunRecord:
        rec.total_ms = (time.perf_counter() - t0) * 1000.0
        return rec
