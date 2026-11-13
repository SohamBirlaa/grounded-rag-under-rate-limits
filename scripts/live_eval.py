"""RAG evals on the REAL Groq model (Table 3 + answer-quality evidence).

  # free: retrieval metrics for every setting, no API calls
  python scripts/live_eval.py --retrieval-only

  # real answers: 20 questions x 3 settings ~= 60 calls, ~90K tokens on openai/gpt-oss-20b (200K/day)
  python scripts/live_eval.py --settings S1_256_3_no,S3_450_8to4_rrf_no,R5_hybrid_route_rerank \
      --judge-model openai/gpt-oss-120b --resume

Outputs: data/live/results.csv rows (scenario=retrieval_eval, source=real), data/live/traces.jsonl,
data/live/eval_summary.json, data/live/eval_review.csv (add your own human_grade column!).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from _live_common import ROOT, build_pipeline, load_env, load_questions, load_retriever, make_budget

ALL_SETTINGS = ["S1_256_3_no", "S3_450_8to4_rrf_no", "R4_hybrid_route", "R5_hybrid_route_rerank",
                "R6_rerank_nocompress", "R7_compact"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


async def retrieval_only(questions, retr, legacy, settings):
    from relay.live.compress import Compressor
    from relay.live.evals import retrieval_metrics
    from relay.live.pipeline import LEGACY_SETTINGS
    from relay.live.retrieval import SETTINGS

    comp = Compressor(c.text for c in retr.chunks)
    table = {}
    for key in settings:
        rows = []
        for q in questions:
            if key in LEGACY_SETTINGS:
                rr = await legacy.retrieve(q["question"], key)
                st = LEGACY_SETTINGS[key]
            else:
                rr = await retr.retrieve(q["question"], SETTINGS[key])
                st = SETTINGS[key]
            ev = comp.build(q["question"], rr.chunks, compress=st.compress)
            m = retrieval_metrics(q, rr.chunks, " ".join(e.text for e in ev))
            m.update(qid=q["question_id"], ms=rr.ms, ctx_tokens_before=sum(e.tokens_before for e in ev),
                     ctx_tokens_after=sum(e.tokens_after for e in ev), top_score=rr.top_rerank_score,
                     unanswerable=q["is_unanswerable"])
            rows.append(m)
        ans = [r for r in rows if not r["unanswerable"]]
        table[key] = {
            "label": st.name, "n": len(rows),
            "hit_at_k": mean([r["hit_at_k"] for r in ans]), "mrr": mean([r["mrr"] for r in ans]),
            "paper_precision": mean([r["paper_precision"] for r in ans]),
            "fact_in_context": mean([r["fact_in_context"] for r in ans]),
            "fact_retention_after_compression": mean([r.get("fact_retention") for r in ans]),
            "ctx_tokens_before": mean([r["ctx_tokens_before"] for r in rows]),
            "ctx_tokens_after": mean([r["ctx_tokens_after"] for r in rows]),
            "retrieval_ms_p50": mean([statistics.median([r["ms"] for r in rows])]),
            "top_score_answerable_min": mean([min([r["top_score"] for r in ans if r["top_score"] is not None] or [None])]) if any(r["top_score"] is not None for r in ans) else None,
            "top_score_unanswerable_max": max([r["top_score"] for r in rows if r["unanswerable"] and r["top_score"] is not None], default=None),
            "per_question": rows,
        }
        t = table[key]
        print(f"{key:24s} hit@k={t['hit_at_k']} mrr={t['mrr']} paper_prec={t['paper_precision']} "
              f"facts_in_ctx={t['fact_in_context']} retained={t['fact_retention_after_compression']} "
              f"ctx_tok {t['ctx_tokens_before']}->{t['ctx_tokens_after']}")
        if t["top_score_answerable_min"] is not None:
            print(f"{'':24s} rerank score: lowest answerable={t['top_score_answerable_min']}, "
                  f"highest unanswerable={t['top_score_unanswerable_max']}  (a safe abstain threshold lies between, if any)")
    return table


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--settings", default="S1_256_3_no,S3_450_8to4_rrf_no,R5_hybrid_route_rerank")
    ap.add_argument("--questions", default=str(ROOT / "data/eval/questions.json"))
    ap.add_argument("--judge-model", default=None, help="e.g. openai/gpt-oss-120b (separate limit pool)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/live/results.csv"))
    ap.add_argument("--traces", default=str(ROOT / "data/live/traces.jsonl"))
    ap.add_argument("--prompt-profile", default="lean", choices=["lean", "cache_friendly", "auto"])
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    questions = load_questions(Path(a.questions))
    retr, legacy = load_retriever(with_legacy=True)
    retr.warmup(rerank=True)

    rt = await retrieval_only(questions, retr, legacy, ALL_SETTINGS if a.retrieval_only else a.settings.split(","))
    out_dir = Path(a.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_metrics.json").write_text(json.dumps(rt, indent=1, default=str), encoding="utf-8")
    if a.retrieval_only:
        print("wrote", out_dir / "retrieval_metrics.json")
        return

    load_env()
    settings = a.settings.split(",")
    n_calls = len(settings) * len(questions)
    print(f"About to make ~{n_calls} REAL generation calls (+repairs{', + judge calls' if a.judge_model else ''}).")
    if not a.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return

    from relay.live.evals import grade_answer, judge_messages, parse_judge
    from relay.live.groq_client import GroqClient
    from relay.live.records import ResultsWriter

    writer = ResultsWriter(Path(a.out), Path(a.traces))
    done = writer.done_ids() if a.resume else set()
    review_path = out_dir / "eval_review.csv"
    review_new = not review_path.exists()
    review_f = open(review_path, "a", newline="", encoding="utf-8")
    review = csv.writer(review_f)
    if review_new:
        review.writerow(["request_id", "setting", "question_id", "question", "gold_answer", "answer",
                         "auto_grade", "judge_correct", "judge_reason", "human_grade"])
    summary = defaultdict(list)
    async with GroqClient(capture_dir=ROOT / "data/calibration") as client:
        pipe = build_pipeline(client, a.model, retriever=retr, legacy=legacy, deadline_s=30.0, prompt_profile=a.prompt_profile,
                              use_exact_cache=False, use_semantic_cache=False)
        if a.judge_model:
            pipe.budgets[a.judge_model] = make_budget(a.judge_model)
        for key in settings:
            for q in questions:
                rid = f"eval_{key}_{q['question_id']}"
                if rid in done:
                    continue
                # Pace OUTSIDE the timed request so Table 3 latency is not polluted by our own pacing.
                while pipe.budgets[a.model].pressure() > 0.45:
                    await asyncio.sleep(1.0)
                rec = await pipe.run(q["question"], key, rid, "retrieval_eval")
                judge = None
                if a.judge_model and rec.status == "ok" and rec.cited_ids:
                    ev = " ".join(retr.by_id[c].text for c in rec.evidence_ids if c in retr.by_id)
                    while pipe.budgets[a.judge_model].pressure() > 0.6:
                        await asyncio.sleep(1.0)
                    jr = await client.stream_chat(a.judge_model, judge_messages(q, rec.answer, ev), 500,
                                                  budget=pipe.budgets[a.judge_model])
                    judge = parse_judge(jr.text) if jr.ok else None
                ev_ = grade_answer(q, rec.answer, rec.cited_ids, rec.status, rec.gate_emitted, rec.gate_rejected, judge)
                writer.write(rec, ev_.grade, {"eval": ev_.__dict__, "question_id": q["question_id"]})
                review.writerow([rid, rec.setting_label, q["question_id"], q["question"], q["gold_answer"], rec.answer,
                                 ev_.grade, None if not judge else judge["correct"], None if not judge else judge["reason"], ""])
                review_f.flush()
                summary[key].append(ev_)
                print(f"{rid:40s} {rec.status:12s} {ev_.grade:18s} ttft={rec.first_token_ms or 0:7.0f}ms "
                      f"tok={rec.tokens_in}+{rec.tokens_out} path={rec.path} recall={ev_.fact_recall}")
    writer.close()
    review_f.close()
    agg = {}
    for key, evs in summary.items():
        graded = [e for e in evs if e.grade]
        agg[key] = {
            "graded": len(graded),
            **{g: round(100 * sum(e.grade == g for e in graded) / max(1, len(graded)), 1)
               for g in ("correct", "wrong", "refused_correctly", "refused_wrongly")},
            "fact_recall": mean([e.fact_recall for e in graded]),
            "faithfulness": mean([e.faithfulness for e in graded]),
            "judge_agreement_with_rules": mean([float(e.judge_correct == (e.grade == "correct"))
                                                for e in graded if e.judge_correct is not None]),
        }
    (out_dir / "eval_summary.json").write_text(json.dumps({"generation": agg, "retrieval": {
        k: {kk: vv for kk, vv in v.items() if kk != "per_question"} for k, v in rt.items()}}, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))
    print("Now open data/live/eval_review.csv and fill human_grade yourself for at least 15 rows.")


if __name__ == "__main__":
    asyncio.run(main())
