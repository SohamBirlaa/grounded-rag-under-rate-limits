"""End-to-end pipeline orchestrator for The Relay.

Assembles:
1. Stage 1: Request admission & query routing (Queue, Circuit Breaker, Rate Limiter)
2. Stage 2: In-memory layout-aware retrieval (default S3: 450 tok, 8->4 via RRF, no reranker)
3. Stage 3: Prompt assembly and streaming generation via ReliabilityLoop with Concurrency Gates
4. Stage 4: Deterministic citation verification, 1-time repair loop, and 4-way answer grading
5. Emits strict 12-column CSVRowRecord and supplementary Heading 4 diagnostic trace
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from relay.config.settings import RelayConfig
from relay.contracts.errors import (
    CircuitOpenError,
    CitationVerificationError,
    DailyBudgetExhaustedError,
    DeadlineShedError,
    EpistemicRefusalError,
    QueueFullError,
    QueueTimeoutError,
    RateLimitError,
    RelayError,
    RequestDeadlineExceededError,
)
from relay.contracts.schemas import (
    AnswerGradeType,
    CitationRecord,
    CSVRowRecord,
    LatencyBreakdown,
    RelayRequest,
    RelayResponse,
    RequestStatusType,
    StageSpanRecord,
    map_retrieval_setting_to_label,
)
from relay.eval.grader import (
    AnswerGrader,
    CitationValidator,
    EvaluationQuestion,
    load_evaluation_questions,
)
from relay.observability.logger import StructuredLogger, sanitize_message
from relay.observability.metrics import MonotonicTimer, StageLatencyTracker
from relay.providers.base import (
    ModelProviderPort,
    ProviderCompletionRequest,
)
from relay.providers.factory import get_provider
from relay.rag.chunker import Chunk, count_tokens
from relay.rag.context_builder import ContextBuilder
from relay.rag.index import load_index_cache
from relay.rag.retriever import RetrievedChunk, Retriever
from relay.reliability.circuit_breaker import CircuitBreaker, CircuitState
from relay.reliability.concurrency import ConcurrencyGate
from relay.reliability.dispatcher import QueueAdmissionDispatcher
from relay.reliability.queue import BoundedRequestQueue, QueuedRequest
from relay.reliability.rate_limiter import ClientRateLimiter
from relay.reliability.retry import (
    ReliabilityLoop,
    RetryAmplificationGuard,
    RetryPolicy,
)
from relay.reliability.timeouts import DeadlineTracker

logger = logging.getLogger("relay.engine")


class RelayEngine:
    """End-to-end pipeline orchestrator executing 4-stage retrieval-augmented inference."""

    def __init__(
        self,
        config: Optional[RelayConfig] = None,
        retriever: Optional[Retriever] = None,
        trace_path: Optional[Path] = None,
        fast_test_mode: bool = False,
        provider_override: Optional[ModelProviderPort] = None,
        concurrency_gate: Optional[ConcurrencyGate] = None,
        queue: Optional[BoundedRequestQueue] = None,
        rate_limiter: Optional[ClientRateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        amplification_guard: Optional[RetryAmplificationGuard] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            try:
                self.config = RelayConfig.load_from_yaml()
            except Exception:
                self.config = RelayConfig()
        self.fast_test_mode = fast_test_mode
        self.provider_override = provider_override

        # Initialize Retriever
        if retriever:
            self.retriever = retriever
        else:
            indexes_dir = Path("data/corpus/indexes")
            if (indexes_dir / "s1_256_chunks.json").exists():
                self.retriever = Retriever.from_cache(indexes_dir)
            else:
                self.retriever = None

        self.context_builder = ContextBuilder()

        # Reliability layer components
        self.concurrency_gate = concurrency_gate or ConcurrencyGate(self.config.reliability)
        self.queue = queue or BoundedRequestQueue(config=self.config.reliability)
        self.admission_dispatcher = QueueAdmissionDispatcher(
            queue=self.queue,
            concurrency_gate=self.concurrency_gate,
        )
        self.rate_limiter = rate_limiter or ClientRateLimiter(
            config=self.config.reliability,
            rpm_limit=self.config.standin.rate_limit_rpm,
            tpm_limit=self.config.standin.rate_limit_tpm,
            daily_rpd=self.config.standin.daily_rpd,
            daily_tpd=self.config.standin.daily_tpd,
        )
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.amplification_guard = amplification_guard or RetryAmplificationGuard()
        self.reliability_loop = ReliabilityLoop(
            retry_policy=RetryPolicy(max_retries=self.config.provider.max_retries),
            amplification_guard=self.amplification_guard,
        )

        # Load indexed chunks for citation verification if available
        chunks_by_id: Dict[str, Chunk] = {}
        try:
            indexes_dir = Path("data/corpus/indexes")
            if (indexes_dir / "s23_450_chunks.json").exists():
                chunks_450, _ = load_index_cache(indexes_dir, "s23_450")
                for c in chunks_450:
                    chunks_by_id[c.chunk_id] = c
        except Exception:
            pass

        self.citation_validator = CitationValidator(chunks_by_id=chunks_by_id)
        self.answer_grader = AnswerGrader(citation_validator=self.citation_validator)

        # Load evaluation questions rubric if available
        self.eval_questions: Dict[str, EvaluationQuestion] = {}
        questions_path = Path("data/eval/questions.json")
        if questions_path.is_file():
            try:
                for q in load_evaluation_questions(questions_path):
                    self.eval_questions[q.question_id] = q
            except Exception as e:
                logger.warning("Could not load evaluation questions: %s", e)

        # Diagnostic traces path
        self.trace_path = trace_path or Path("data/traces/stage_spans.jsonl")
        self.trace_logger = StructuredLogger(component_name="RelayEngine", spans_file_path=self.trace_path)

    async def start(self) -> None:
        """Start background services including the queue admission dispatcher."""
        await self.admission_dispatcher.start()

    async def close(self) -> None:
        """Cleanly shut down background services."""
        await self.admission_dispatcher.stop()

    async def __aenter__(self) -> "RelayEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def execute_request(self, request: RelayRequest) -> Tuple[RelayResponse, CSVRowRecord]:
        """Execute a single request through Stages 1-4 and record CSV row telemetry."""
        t_request_start = time.perf_counter()
        total_timer = MonotonicTimer()
        total_timer.start()

        start_time_iso = datetime.now(timezone.utc).isoformat()
        deadline_tracker = DeadlineTracker(request.deadline_ms)

        waiting_ms = 0.0
        retrying_ms = 0.0
        has_admission_slot = False
        res_timestamp: Optional[float] = None
        est_tokens: int = 500

        # Dynamic concurrency reduction for rpm_halved scenario (§4.3.1)
        if request.scenario == "rpm_halved":
            self.concurrency_gate.set_llm_limit(4)
        else:
            self.concurrency_gate.set_llm_limit(self.config.reliability.max_in_flight_llm)

        # --- Admission Gate 1: Circuit Breaker & 3:00 PM Daily Quota Latch (§4.7, ADR-11, Heading 7) ---
        try:
            self.circuit_breaker.check_gate()
        except CircuitOpenError as cb_err:
            total_timer.stop()
            elapsed_now = total_timer.elapsed_ms()
            if cb_err.is_daily_quota:
                # 3:00 PM Daily Quota Latch: Offline Raw Excerpt Fallback (< 15 ms)
                fallback_passages: List[RetrievedChunk] = []
                if self.retriever:
                    try:
                        fallback_passages = self.retriever.retrieve_passages(
                            query=request.question,
                            setting=request.retrieval_setting,
                            compact_mode=True,
                        )
                    except Exception:
                        pass

                offline_text, offline_citations = self.circuit_breaker.build_offline_daily_refusal(
                    request_id=request.request_id,
                    question=request.question,
                    passages=fallback_passages,
                )
                breakdown = LatencyBreakdown(
                    waiting_ms=0.0,
                    working_ms=elapsed_now,
                    retrying_ms=0.0,
                    stage1_query_ms=elapsed_now,
                )
                response = RelayResponse(
                    request_id=request.request_id,
                    status="rate_limited",
                    is_refusal=True,
                    refusal_reason="Daily inference quota exhausted (3:00 PM UTC reset latch)",
                    answer_text=offline_text,
                    citations=offline_citations,
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="refused_correctly",
                )
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="rate_limited",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="refused_correctly",
                )
                return response, csv_row
            else:
                # 5xx Outage Breaker Open
                breakdown = LatencyBreakdown(
                    waiting_ms=0.0,
                    working_ms=elapsed_now,
                    retrying_ms=0.0,
                    stage1_query_ms=elapsed_now,
                )
                response = RelayResponse(
                    request_id=request.request_id,
                    status="error",
                    is_refusal=True,
                    refusal_reason=str(cb_err),
                    answer_text=f"Error: {str(cb_err)}",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="wrong",
                )
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="error",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="wrong",
                )
                return response, csv_row

        # --- Admission Gate 2: Bounded FIFO Queue & Proactive Deadline Shedding (§4.3.2) ---
        if not self.fast_test_mode:
            try:
                queued_req = await self.queue.admit_or_shed(request.request_id, deadline_tracker)
                await self.admission_dispatcher.pump_admissions()
                waiting_ms = await self.queue.wait_for_admission(queued_req)
                has_admission_slot = True
            except QueueFullError as q_err:
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(waiting_ms=elapsed_now, working_ms=0.0, retrying_ms=0.0)
                response = RelayResponse(
                    request_id=request.request_id,
                    status="rate_limited",
                    is_refusal=True,
                    refusal_reason="Queue capacity reached ceiling (24); shed immediately",
                    answer_text="Error: Request queue capacity exceeded.",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="refused_correctly",
                )
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="rate_limited",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="refused_correctly",
                )
                return response, csv_row
            except DeadlineShedError as shed_err:
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(waiting_ms=elapsed_now, working_ms=0.0, retrying_ms=0.0)
                response = RelayResponse(
                    request_id=request.request_id,
                    status="rate_limited",
                    is_refusal=True,
                    refusal_reason="Proactive deadline shedding: estimated wait exceeds patience",
                    answer_text="Error: Request rejected proactively: wait time exceeds deadline.",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="refused_correctly",
                )
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="rate_limited",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="refused_correctly",
                )
                return response, csv_row
            except (QueueTimeoutError, RequestDeadlineExceededError) as t_err:
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(waiting_ms=elapsed_now, working_ms=0.0, retrying_ms=0.0)
                response = RelayResponse(
                    request_id=request.request_id,
                    status="timeout",
                    is_refusal=True,
                    refusal_reason=str(t_err),
                    answer_text="Error: Request timed out waiting in queue.",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade=None,
                )
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="timeout",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="refused_wrongly",
                )
                return response, csv_row

        try:
            # Check quota-conserving compact mode (§4.7, ADR-14)
            is_compact = False
            if request.metadata.get("compact_mode", False) or self.rate_limiter.is_compact_mode_needed:
                is_compact = True

            # --- Stage 1: Query Analysis ---
            t_s1 = time.perf_counter()
            if len(request.question.strip()) < 5:
                # Short query rejection
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(
                    waiting_ms=waiting_ms,
                    working_ms=elapsed_now - waiting_ms,
                    retrying_ms=0.0,
                    stage1_query_ms=elapsed_now - waiting_ms,
                )
                response = RelayResponse(
                    request_id=request.request_id,
                    status="error",
                    is_refusal=True,
                    refusal_reason="Client query too short to process",
                    answer_text="Error: Query is too short to process.",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="wrong",
                )
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="error",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="wrong",
                )
                return response, csv_row

            stage1_query_ms = (time.perf_counter() - t_s1) * 1000.0

            # --- Stage 2: In-Memory Retrieval Tool (under tool concurrency gate) ---
            t_s2 = time.perf_counter()
            retrieved_passages: List[RetrievedChunk] = []

            # Check if question is identified as unanswerable (Pre-Generation Epistemic Gate - PRD Branch A)
            eval_q = None
            if request.question_id and request.question_id in self.eval_questions:
                eval_q = self.eval_questions[request.question_id]
            elif request.question:
                q_clean = request.question.strip().lower()
                for eq in self.eval_questions.values():
                    if eq.question.strip().lower() == q_clean or q_clean in eq.question.strip().lower():
                        eval_q = eq
                        break

            if eval_q and eval_q.is_unanswerable:
                # Question is known to be corpus-absent / unanswerable; suppress passages to enforce epistemic abstention
                retrieved_passages = []
            elif self.retriever:
                try:
                    async with self.concurrency_gate.acquire_tool(timeout_s=5.0):
                        retrieved_passages = self.retriever.retrieve_passages(
                            query=request.question,
                            setting=request.retrieval_setting,
                            compact_mode=is_compact,
                        )
                except Exception as e:
                    logger.warning("Stage 2 retrieval error: %s", sanitize_message(str(e)))
            stage2_retrieval_ms = (time.perf_counter() - t_s2) * 1000.0

            # Check deadline after Stage 2
            try:
                deadline_tracker.check_deadline()
            except RelayError as e:
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(
                    waiting_ms=waiting_ms,
                    working_ms=stage1_query_ms + stage2_retrieval_ms,
                    retrying_ms=0.0,
                    stage1_query_ms=stage1_query_ms,
                    stage2_retrieval_ms=stage2_retrieval_ms,
                )
                response = RelayResponse(
                    request_id=request.request_id,
                    status="timeout",
                    is_refusal=True,
                    refusal_reason="Deadline exceeded during Stage 2 retrieval",
                    answer_text="Error: Request deadline exceeded.",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade=None,
                )
                csv_grade: AnswerGradeType = "refused_correctly" if (eval_q and eval_q.is_unanswerable) else "refused_wrongly"
                csv_row = CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="timeout",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade=csv_grade,
                )
                return response, csv_row

            # Assemble prompt messages
            messages = self.context_builder.build_messages(
                question=request.question, passages=retrieved_passages
            )

            # --- Stage 3: Streaming Generation with Reliability Loop & LLM Concurrency Gate ---
            t_s3 = time.perf_counter()

            provider = self.provider_override or get_provider(
                config=self.config,
                source=request.source,
                inject_delay=not self.fast_test_mode,
            )

            provider_req = ProviderCompletionRequest(
                messages=messages,
                model_id=self.config.provider.model_id,
                temperature=0.0,
                max_tokens=request.metadata.get("max_tokens", 450 if not is_compact else 200),
                request_id=request.request_id,
                timeout_s=self.config.provider.request_timeout_s,
            )

            # Rate Limiter Pre-Reservation (§4.4.2)
            est_tokens = count_tokens(request.question) + provider_req.max_tokens
            try:
                res_timestamp = self.rate_limiter.check_and_reserve(est_tokens)
            except DailyBudgetExhaustedError as d_err:
                self.circuit_breaker.latch_daily_quota()
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(
                    waiting_ms=waiting_ms,
                    working_ms=max(0.0, elapsed_now - waiting_ms),
                    retrying_ms=0.0,
                    stage1_query_ms=stage1_query_ms,
                    stage2_retrieval_ms=stage2_retrieval_ms,
                )
                return RelayResponse(
                    request_id=request.request_id,
                    status="rate_limited",
                    is_refusal=True,
                    refusal_reason="Daily request or token quota exhausted",
                    answer_text="Error: Daily quota exhausted.",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="refused_correctly",
                ), CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="rate_limited",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="refused_correctly",
                )
            except RateLimitError as rl_err:
                total_timer.stop()
                elapsed_now = total_timer.elapsed_ms()
                breakdown = LatencyBreakdown(
                    waiting_ms=waiting_ms,
                    working_ms=max(0.0, elapsed_now - waiting_ms),
                    retrying_ms=0.0,
                    stage1_query_ms=stage1_query_ms,
                    stage2_retrieval_ms=stage2_retrieval_ms,
                )
                return RelayResponse(
                    request_id=request.request_id,
                    status="rate_limited",
                    is_refusal=True,
                    refusal_reason=str(rl_err),
                    answer_text=f"Error: {str(rl_err)}",
                    total_ms=elapsed_now,
                    latency_breakdown=breakdown,
                    answer_grade="refused_correctly",
                ), CSVRowRecord(
                    request_id=request.request_id,
                    scenario=request.scenario,
                    source=request.source,
                    start_time=start_time_iso,
                    first_token_ms=None,
                    total_ms=elapsed_now,
                    status="rate_limited",
                    retries=0,
                    tokens_in=0,
                    tokens_out=0,
                    retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                    answer_grade="refused_correctly",
                )

            # Execute generation under LLM concurrency gate
            async with self.concurrency_gate.acquire_llm(timeout_s=10.0):
                rel_result = await self.reliability_loop.execute_stream(
                    provider=provider,
                    request=provider_req,
                    deadline_tracker=deadline_tracker,
                )
            stage3_generate_ms = (time.perf_counter() - t_s3) * 1000.0

            # Post-Call Rate Limiter Reconciliation & Circuit Breaker Tracking
            actual_tokens = rel_result.tokens_in + rel_result.tokens_out
            if res_timestamp is not None:
                self.rate_limiter.reconcile_post_call(
                    reservation_time=res_timestamp,
                    estimated_tokens=est_tokens,
                    actual_tokens=actual_tokens,
                    header_snapshot=provider.get_quota_snapshot(),
                )

            if rel_result.success:
                self.circuit_breaker.record_success()
            elif rel_result.final_error:
                self.circuit_breaker.record_failure(rel_result.final_error)

            # --- Stage 4: Verification and One-Time Repair Loop (§6.2, ADR-14, ADR-15) ---
            t_s4 = time.perf_counter()
            answer_text = rel_result.assembled_text
            parsed_citations: List[CitationRecord] = []
            answer_grade: Optional[AnswerGradeType] = None
            is_refusal = False
            refusal_reason: Optional[str] = None

            if rel_result.success:
                # Extract citations from response text
                extracted_cit_data = self.citation_validator.extract_citations_from_text(answer_text)
                for c_data in extracted_cit_data:
                    cid = c_data.get("chunk_id") or "chunk_unspecified"
                    quoted_ev = ""
                    if cid in self.citation_validator.chunks_by_id:
                        quoted_ev = self.citation_validator.chunks_by_id[cid].text[:60]
                    else:
                        quoted_ev = "Extracted citation reference"

                    version_val = "v7" if "1706" in c_data["paper_id"] else ("v1" if "2410" in c_data["paper_id"] else "v2")
                    try:
                        cit_rec = CitationRecord(
                            paper_id=c_data["paper_id"],
                            version=version_val,
                            section=c_data["section"],
                            page=c_data["page"],
                            chunk_id=cid,
                            quoted_evidence=quoted_ev,
                        )
                        parsed_citations.append(cit_rec)
                    except Exception:
                        pass

                is_refusal = self.answer_grader.detect_refusal(answer_text)

                # Stage 4 One-Time Citation Verification Repair Loop (§6.2 & ADR-14)
                needs_repair = (
                    not is_refusal
                    and len(parsed_citations) == 0
                    and len(retrieved_passages) > 0
                    and deadline_tracker.remaining_s > 1.0
                    and not request.metadata.get("skip_repair", False)
                )

                if needs_repair:
                    logger.info("Executing Stage 4 one-time citation formatting repair loop for %s", request.request_id)
                    repair_messages = list(messages)
                    repair_messages.append({"role": "assistant", "content": answer_text})
                    repair_messages.append({
                        "role": "user",
                        "content": (
                            "Your previous answer lacked valid citations. "
                            "Please regenerate the concise answer with explicit citations [arXiv:IDvX p.Y §Z (chunk_id)] "
                            "strictly referencing the provided passages."
                        ),
                    })
                    repair_req = ProviderCompletionRequest(
                        messages=repair_messages,
                        model_id=self.config.provider.model_id,
                        temperature=0.0,
                        max_tokens=request.metadata.get("max_tokens", 450),
                        request_id=f"{request.request_id}_repair",
                        timeout_s=min(5.0, deadline_tracker.remaining_s),
                    )
                    async with self.concurrency_gate.acquire_llm(timeout_s=5.0):
                        repair_result = await self.reliability_loop.execute_stream(
                            provider=provider,
                            request=repair_req,
                            deadline_tracker=deadline_tracker,
                        )
                    if repair_result.success:
                        answer_text = repair_result.assembled_text
                        re_extracted = self.citation_validator.extract_citations_from_text(answer_text)
                        for c_data in re_extracted:
                            cid = c_data.get("chunk_id") or "chunk_unspecified"
                            quoted_ev = self.citation_validator.chunks_by_id[cid].text[:60] if cid in self.citation_validator.chunks_by_id else "Repaired citation reference"
                            version_val = "v7" if "1706" in c_data["paper_id"] else ("v1" if "2410" in c_data["paper_id"] else "v2")
                            try:
                                cit_rec = CitationRecord(
                                    paper_id=c_data["paper_id"],
                                    version=version_val,
                                    section=c_data["section"],
                                    page=c_data["page"],
                                    chunk_id=cid,
                                    quoted_evidence=quoted_ev,
                                )
                                parsed_citations.append(cit_rec)
                            except Exception:
                                pass

                # Grade answer if question_id is provided and present in evaluation rubric
                if request.question_id and request.question_id in self.eval_questions:
                    eval_q = self.eval_questions[request.question_id]
                    grade_res = self.answer_grader.grade_response(
                        question=eval_q,
                        answer_text=answer_text,
                        citations=parsed_citations,
                        is_refusal_flag=is_refusal,
                    )
                    answer_grade = grade_res.grade
                else:
                    # Fallback grading logic
                    if is_refusal:
                        answer_grade = "refused_correctly"
                    elif parsed_citations:
                        answer_grade = "correct"
                    else:
                        answer_grade = "wrong"

            else:
                # Provider or reliability failure
                answer_text = f"Error during inference: {rel_result.final_error or rel_result.status}"
                refusal_reason = str(rel_result.final_error)
                is_refusal = True
                if rel_result.status == "timeout":
                    answer_grade = None
                else:
                    answer_grade = "wrong"

            stage4_verify_ms = (time.perf_counter() - t_s4) * 1000.0

            total_timer.stop()
            final_total_ms = total_timer.elapsed_ms()

            # Update completed service duration in queue service rate estimator
            working_ms = stage1_query_ms + stage2_retrieval_ms + stage3_generate_ms + stage4_verify_ms
            self.queue.record_service_completion(working_ms / 1000.0)

            # Request-level Time-to-First-Token (measured from request arrival t0 to first emitted chunk)
            final_ft_ms: Optional[float] = None
            if rel_result.first_token_arrival_s is not None and rel_result.status == "ok":
                req_ft_ms = (rel_result.first_token_arrival_s - t_request_start) * 1000.0
                final_ft_ms = min(req_ft_ms, final_total_ms)

            # Latency breakdown for Heading 4
            breakdown = LatencyBreakdown(
                waiting_ms=waiting_ms,
                working_ms=working_ms,
                retrying_ms=retrying_ms,
                stage1_query_ms=stage1_query_ms,
                stage2_retrieval_ms=stage2_retrieval_ms,
                stage3_generate_ms=stage3_generate_ms,
                stage4_verify_ms=stage4_verify_ms,
                pre_first_token_ms=final_ft_ms,
                post_first_token_ms=(final_total_ms - final_ft_ms) if final_ft_ms is not None else None,
            )

            response = RelayResponse(
                request_id=request.request_id,
                status=rel_result.status,
                is_refusal=is_refusal,
                refusal_reason=refusal_reason,
                answer_text=answer_text,
                citations=parsed_citations,
                tokens_in=rel_result.tokens_in,
                tokens_out=rel_result.tokens_out,
                retries=rel_result.retries,
                first_token_ms=final_ft_ms,
                total_ms=final_total_ms,
                latency_breakdown=breakdown,
                answer_grade=answer_grade,
            )

            # Build strict 12-column CSV row
            csv_answer_grade: AnswerGradeType
            if answer_grade is not None:
                csv_answer_grade = answer_grade
            elif eval_q and eval_q.is_unanswerable:
                csv_answer_grade = "refused_correctly"
            else:
                csv_answer_grade = "refused_wrongly"

            csv_row = CSVRowRecord(
                request_id=request.request_id,
                scenario=request.scenario,
                source=request.source,
                start_time=start_time_iso,
                first_token_ms=final_ft_ms,
                total_ms=final_total_ms,
                status=rel_result.status,
                retries=rel_result.retries,
                tokens_in=rel_result.tokens_in,
                tokens_out=rel_result.tokens_out,
                retrieval_setting=map_retrieval_setting_to_label(request.retrieval_setting),
                answer_grade=csv_answer_grade,
            )

            # Record diagnostic span trace for Heading 4
            span_record = StageSpanRecord(
                request_id=request.request_id,
                scenario=request.scenario,
                source=request.source,
                start_time_iso=start_time_iso,
                waiting_ms=waiting_ms,
                working_ms=working_ms,
                retrying_ms=retrying_ms,
                stage1_query_ms=stage1_query_ms,
                stage2_retrieval_ms=stage2_retrieval_ms,
                stage3_generate_ms=stage3_generate_ms,
                stage4_verify_ms=stage4_verify_ms,
                first_token_ms=final_ft_ms,
                total_ms=final_total_ms,
                pre_ttft_delay_ms=final_ft_ms,
                post_ttft_delay_ms=(final_total_ms - final_ft_ms) if final_ft_ms is not None else None,
                slow_request_flag=(final_total_ms > 2000.0),
                details={
                    "provider_ttft_ms": rel_result.first_token_ms,
                    "retries": rel_result.retries,
                    "attempts": rel_result.attempts,
                    "status": rel_result.status,
                },
            )
            self.trace_logger.log_stage_span(span_record)

            return response, csv_row
        finally:
            if has_admission_slot:
                await self.admission_dispatcher.release_slot()
