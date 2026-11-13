"""Load shapes against the REAL Groq service (Tables 2a/2b). No stand-in.

Our admission control keeps what we SEND within Groq's limits; requests beyond capacity are
queued (if they can start before the person's deadline) or shed. That behaviour is the result.

Scenarios (approx real requests offered; on openai/gpt-oss-20b: 8K TPM, 1K req/day, 200K tok/day):
  steady              4/min for 5 min                       ~20 sent, ~26K tok
  burst               15 arrivals within 5 s                ~15 offered, ~6-8 admitted
  sustained_overload  20/min for 4 min                      ~80 offered, ~15-20 admitted
  rpm_halved          our budget set to 50% of the limits, 8/min for 3 min
  daily_budget        app daily cap of 5 requests, then 10 more arrive -> daily-budget path
                      (for the REAL per-day 429 use `live_probe.py --phases daily` as the last run.)
  cache_repeat        repeated + paraphrased questions, 6/min for 3 min -> exact/semantic cache hits
  naive_retry         like overload but admission OFF + naive client retries for 2 min:
                      measures on the real service how retrying makes things worse
Run one at a time:  python scripts/live_load.py --scenario steady
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

from _live_common import ROOT, build_pipeline, load_env, load_questions, load_retriever

SCENARIOS = {
    "steady": dict(rate_per_min=4, duration_s=300),
    "burst": dict(burst=15, burst_window_s=5),
    "sustained_overload": dict(rate_per_min=20, duration_s=240),
    "rpm_halved": dict(rate_per_min=8, duration_s=180, budget_scale=0.5),
    "daily_budget": dict(rate_per_min=6, duration_s=150, local_daily_cap=5),
    "cache_repeat": dict(rate_per_min=6, duration_s=180, pool="repeat"),
    "naive_retry": dict(rate_per_min=20, duration_s=120, naive=True),
}

# Real-world repeats: same question, reworded (semantic cache should hit) or changed in a way that
# changes the answer (base vs big: the discriminator guard must MISS).
REPEAT_POOL = [
    "What is the frame rate of the Mimi codec in Moshi?",
    "At what frame rate does Moshi's Mimi codec operate?",
    "Mimi codec frame rate in the Moshi paper?",
    "In 'Attention Is All You Need', what is d_model in the base Transformer?",
    "What is d_model for the base model in Attention Is All You Need?",
    "In 'Attention Is All You Need', what is d_model in the big Transformer?",
]


def arrivals(spec, seed: int):
    rnd = random.Random(seed)
    if "burst" in spec:
        return sorted(rnd.uniform(0, spec["burst_window_s"]) for _ in range(spec["burst"]))
    t, out, lam = 0.0, [], spec["rate_per_min"] / 60.0
    while True:
        t += rnd.expovariate(lam)  # Poisson arrivals: real traffic is not evenly spaced
        if t > spec["duration_s"]:
            return out
        out.append(t)


ORDER = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_budget", "cache_repeat", "naive_retry"]


def scaled(spec: dict, scale: float) -> dict:
    """Shrink durations / burst size to save daily quota (rates stay the same, so the load SHAPE is kept)."""
    s = dict(spec)
    if "duration_s" in s:
        s["duration_s"] = max(45, int(s["duration_s"] * scale))
    if "burst" in s:
        s["burst"] = max(8, int(s["burst"] * scale))
    return s


async def run_scenario(name, spec, a, client_factory, retr, questions, writer):
    from relay.live.evals import grade_answer

    times = arrivals(spec, a.seed)
    print(f"\n=== {name}: {len(times)} real requests offered over {times[-1] if times else 0:.0f}s on {a.model} ===")
    by_text = {q["question"]: q for q in questions}
    pool = REPEAT_POOL if spec.get("pool") == "repeat" else [q["question"] for q in questions]
    rnd = random.Random(a.seed + 1)
    naive = spec.get("naive", False)
    async with client_factory("naive" if naive else "guarded") as client:
        pipe = build_pipeline(client, a.model, fallback=a.fallback_model, retriever=retr,
                              budget_scale=spec.get("budget_scale", 1.0), local_daily_cap=spec.get("local_daily_cap"),
                              deadline_s=a.deadline_s, prompt_profile=a.prompt_profile,
                              max_in_flight=(64 if naive else a.max_in_flight))
        if naive:  # admission control off: send everything immediately
            for b in pipe.budgets.values():
                b.tpm, b.rpm, b.safety = 10**9, 10**9, 1.0
                b.observe = lambda snap: None          # ignore provider headers too
                b.observe_429 = lambda *args: None
        t_start = time.perf_counter()
        results = []

        async def fire(i: int, at: float):
            await asyncio.sleep(max(0.0, at - (time.perf_counter() - t_start)))
            qtext = rnd.choice(pool)
            rec = await pipe.run(qtext, a.setting, f"{name}_{a.seed}_{i:03d}", name)
            q = by_text.get(qtext)
            grade = grade_answer(q, rec.answer, rec.cited_ids, rec.status).grade if q else ""
            writer.write(rec, grade, {"peak_waiting_so_far": pipe.peak_waiting, "in_flight_at_end": pipe.in_flight})
            results.append(rec)
            print(f"  {rec.request_id} +{at:6.1f}s {rec.status:12s} {rec.path:18s} first={rec.first_token_ms or 0:7.0f}ms "
                  f"total={rec.total_ms:7.0f}ms wait={rec.queue_wait_ms:6.0f}ms retries={rec.retries} "
                  f"tok={rec.tokens_in}+{rec.tokens_out} cached={rec.cached_tokens}")

        await asyncio.gather(*[fire(i, t) for i, t in enumerate(times)])
        ok = [r for r in results if r.status == "ok"]
        day = [r for r in results if r.path == "daily_budget"]
        print(f"done {name}: {len(ok)}/{len(results)} ok, peak waiting={pipe.peak_waiting}, "
              f"semantic cache={pipe.semantic.stats}, 429 captures={len(client.captured_429)}")
        return bool(day) and name != "daily_budget"  # real per-day limit reached -> stop the batch


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS) + ["all"])
    ap.add_argument("--scale", type=float, default=1.0, help="e.g. 0.5 halves durations to save quota")
    ap.add_argument("--gap-s", type=float, default=75.0, help="pause between scenarios with --scenario all")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--fallback-model", default=None)
    ap.add_argument("--setting", default="R5_hybrid_route_rerank")
    ap.add_argument("--deadline-s", type=float, default=12.0, help="how long a person waits before giving up")
    ap.add_argument("--max-in-flight", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data/live/results.csv"))
    ap.add_argument("--traces", default=str(ROOT / "data/live/traces.jsonl"))
    ap.add_argument("--prompt-profile", default="lean", choices=["lean", "cache_friendly", "auto"])
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    names = ORDER if a.scenario == "all" else [a.scenario]
    specs = {n: scaled(SCENARIOS[n], a.scale) for n in names}
    total = sum(len(arrivals(specs[n], a.seed)) for n in names)
    print(f"Scenarios {names} at scale {a.scale}: {total} real requests offered in total.")
    load_env()
    if not a.yes and input("Continue? [y/N] ").strip().lower() != "y":
        return

    from relay.live.groq_client import GroqClient
    from relay.live.records import ResultsWriter

    questions = load_questions()
    retr, _ = load_retriever()
    retr.warmup(rerank=True)
    writer = ResultsWriter(Path(a.out), Path(a.traces))
    done = writer.done_ids()

    def client_factory(policy):
        return GroqClient(capture_dir=ROOT / "data/calibration", retry_policy=policy)

    for k, name in enumerate(names):
        if any(rid.startswith(f"{name}_{a.seed}_") for rid in done):
            print(f"skip {name}: already in {a.out}")
            continue
        stop = await run_scenario(name, specs[name], a, client_factory, retr, questions, writer)
        if stop:
            print("Groq's REAL per-day limit was reached: stopping. Rerun the same command later to continue.")
            break
        if k < len(names) - 1:
            await asyncio.sleep(a.gap_s)  # let the per-minute bucket refill between shapes
    writer.close()


if __name__ == "__main__":
    asyncio.run(main())
